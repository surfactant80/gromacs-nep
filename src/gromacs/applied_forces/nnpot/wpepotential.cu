/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 */
#include "gmxpre.h"

#include "wpepotential.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/stringutil.h"

#include "wpeloader.h"

namespace gmx
{

namespace
{

[[noreturn]] void throwEngineError(const WpeStatus status,
                                   const char*    what,
                                   const std::string& libraryPath)
{
    std::string message = formatString("Wisevolve Potential Engine call '%s' failed: %s",
                                       what,
                                       WpeLoader::statusName(status));
    message += formatString(" (library: %s).", libraryPath.c_str());
    if (status == WPE_ERR_UNSUPPORTED_MODEL)
    {
        message +=
                " This model or workflow is not supported by the WPE engine. "
                "WPE supports NEP4/NEP4-ZBL, qNEP (PPPM/Ewald), LJ, EAM (Zhou 2004, "
                "Dai 2006) and Tersoff 1989, with the model shape restrictions "
                "l_max_3body = 4 and ANN hidden width <= 256. Use "
                "nnpot-backend = gpumd (or the standalone GPUMD executable) for "
                "models that WPE does not implement.";
    }
    else if (status == WPE_ERR_CUDA)
    {
        message +=
                " The WPE engine needs a visible CUDA device. grompp does not load WPE; "
                "this error can only occur in mdrun. Check the GPU visibility and the "
                "WPE GPU architecture package.";
    }
    else if (status == WPE_ERR_NEIGHBOR_CAPACITY)
    {
        message +=
                " The number of neighbors exceeded the WPE capacity, which usually "
                "indicates an unsupported cell-list configuration (for example a box "
                "that is too small for the requested cutoff).";
    }
    throw InconsistentInputError(message);
}

} // namespace

WpePotential::WpePotential(const std::string&      potentialPath,
                           const int               numberOfAtoms,
                           const WpeStageSettings& settings,
                           const std::string&      libraryHint)
{
    std::string loadError;
    WpeLoader*  loader = WpeLoader::instance(libraryHint, &loadError);
    if (loader == nullptr)
    {
        throw InconsistentInputError(
                formatString("The WPE backend was requested but the engine could not be "
                             "loaded: %s",
                             loadError.c_str()));
    }
    libraryPath_   = loader->path();
    engineName_    = loader->info().name;
    engineVersion_ = loader->info().version;
    engineCitation_ = loader->info().citation;

    const WpeApi& api = loader->api();

    WpeCreateInfo createInfo{};
    createInfo.struct_size    = sizeof(createInfo);
    createInfo.potential_path = potentialPath.c_str();
    createInfo.number_of_atoms = numberOfAtoms;

    WpeCreateResult createResult{};
    createResult.struct_size = sizeof(createResult);

    const WpeStatus createStatus = api.create(&createInfo, &createResult, &context_);
    if (createStatus != WPE_OK)
    {
        context_ = nullptr;
        throwEngineError(createStatus, "wpe_create", libraryPath_);
    }
    if (context_ == nullptr || createResult.struct_size < WPE_CREATE_RESULT_V1_MIN_SIZE)
    {
        throw InconsistentInputError(
                "Wisevolve Potential Engine returned an invalid context or an incompatible "
                "WpeCreateResult record.");
    }

    N1 = 0;
    N2 = numberOfAtoms;
    rc = createResult.cutoff;

    activateStage(numberOfAtoms, settings);
}

WpePotential::~WpePotential()
{
    if (context_ != nullptr)
    {
        if (WpeLoader* loader = WpeLoader::instance({}, nullptr))
        {
            loader->api().destroy(context_);
        }
        context_ = nullptr;
    }
}

void WpePotential::activateStage(const int numberOfAtoms, const WpeStageSettings& settings)
{
    std::string loadError;
    WpeLoader*  loader = WpeLoader::instance({}, &loadError);
    if (loader == nullptr || context_ == nullptr)
    {
        throw InconsistentInputError("WPE stage activation requested without a live engine.");
    }

    WpeStageInfo stage{};
    stage.struct_size      = sizeof(stage);
    stage.stage_kind       = WPE_STAGE_RUN;
    stage.number_of_atoms  = numberOfAtoms;
    stage.ensemble_name    = settings.ensembleName.empty() ? nullptr : settings.ensembleName.c_str();
    stage.pimd_beads       = 0;
    stage.qnep_kspace_method = settings.qnepKspaceMethod;
    stage.qnep_pppm_spacing  = settings.qnepPppmSpacing;
    stage.dftd3_enabled      = settings.dftd3Enabled ? 1u : 0u;
    stage.dftd3_functional   = settings.dftd3Functional.empty() ? nullptr
                                                                : settings.dftd3Functional.c_str();
    stage.dftd3_rc_potential             = settings.dftd3RcPotential;
    stage.dftd3_rc_coordination_number   = settings.dftd3RcCoordinationNumber;

    // GROMACS only needs the 6-component system virial, which is much cheaper
    // for WPE than 9N atomic virials (measured ~30% of the step time on water).
    // nnpot-wpe-atom-virial=yes keeps the atomic path for debugging.
    uint64_t facts = settings.needAtomVirial ? WPE_STAGE_HAS_ATOM_VIRIAL_CONSUMER
                                             : WPE_STAGE_HAS_SYSTEM_VIRIAL_CONSUMER;
    if (settings.needChargeOutput)
    {
        facts |= WPE_STAGE_NEED_CHARGE_OUTPUT;
    }
    if (settings.needBecOutput)
    {
        facts |= WPE_STAGE_NEED_BEC_OUTPUT;
    }
    stage.facts = facts;

    const WpeStatus status = loader->api().activateStage(context_, &stage);
    if (status != WPE_OK)
    {
        throwEngineError(status, "wpe_activate_stage", libraryPath_);
    }

    needChargeOutput_ = settings.needChargeOutput;
    needBecOutput_    = settings.needBecOutput;
    pimdBeads_        = 0;
    atomCount_        = numberOfAtoms;

    if (needChargeOutput_ && charge_.size() != static_cast<size_t>(numberOfAtoms))
    {
        charge_.resize(static_cast<size_t>(numberOfAtoms));
    }
    if (needBecOutput_ && bec_.size() != static_cast<size_t>(9) * numberOfAtoms)
    {
        bec_.resize(static_cast<size_t>(9) * numberOfAtoms);
    }
    needAtomVirial_ = settings.needAtomVirial;
}

void WpePotential::compute(Box&                      box,
                           const GPU_Vector<int>&    type,
                           const GPU_Vector<double>& position,
                           GPU_Vector<double>&       potential,
                           GPU_Vector<double>&       force,
                           GPU_Vector<double>&       virial)
{
    if (context_ == nullptr)
    {
        throw InconsistentInputError("WPE compute called without an initialized engine.");
    }
    const int numberOfAtoms = static_cast<int>(type.size());
    if (numberOfAtoms <= 0 || numberOfAtoms != atomCount_)
    {
        throw InconsistentInputError(
                "WPE compute requires the atom count used for stage activation.");
    }
    if (virial.size() < static_cast<size_t>(numberOfAtoms) * 9)
    {
        throw InconsistentInputError("WPE virial buffer is smaller than 9 * number_of_atoms.");
    }
    if (position.size() < static_cast<size_t>(numberOfAtoms) * 3
        || force.size() < static_cast<size_t>(numberOfAtoms) * 3
        || potential.size() < static_cast<size_t>(numberOfAtoms))
    {
        throw InconsistentInputError("WPE output buffers do not match number_of_atoms.");
    }

    std::string loadError;
    WpeLoader*  loader = WpeLoader::instance({}, &loadError);
    if (loader == nullptr)
    {
        throw InconsistentInputError(
                formatString("WPE engine is no longer available: %s", loadError.c_str()));
    }

    pbcX_ = box.pbc_x != 0;
    pbcY_ = box.pbc_y != 0;
    pbcZ_ = box.pbc_z != 0;

    WpeBinding binding{};
    binding.struct_size    = sizeof(binding);
    binding.number_of_atoms = numberOfAtoms;

    // All surfaces are the buffers that already back the NEP device path, so
    // WPE writes forces/energy/virial in place without staging copies.
    binding.position_device      = position.data();
    binding.type_device          = type.data();
    binding.force_device         = force.data();
    binding.potential_device     = potential.data();
    // WPE always needs the 9N atomic-virial surface (it is also what GPUMD's
    // own adapter binds); the stage fact only controls whether the engine
    // additionally maintains the system-virial reduction. Leaving the total
    // virial to engine-owned scratch keeps the fast path.
    binding.atomic_virial_device = virial.data();
    binding.total_virial_device  = nullptr;

    binding.box_h18_host = box.cpu_h;
    binding.pbc_x_host   = &pbcX_;
    binding.pbc_y_host   = &pbcY_;
    binding.pbc_z_host   = &pbcZ_;

    binding.charge_device = needChargeOutput_ ? charge_.data() : nullptr;
    binding.bec_device    = needBecOutput_ ? bec_.data() : nullptr;

    const WpeApi& api = loader->api();

    WpeStatus status = api.bind(context_, &binding);
    if (status != WPE_OK)
    {
        throwEngineError(status, "wpe_bind", libraryPath_);
    }
    status = api.compute(context_);
    if (status != WPE_OK)
    {
        throwEngineError(status, "wpe_compute", libraryPath_);
    }

    // The engine is expected to use the CUDA default stream (like the other
    // external NEP kernels). GMX_WPE_SYNC=1 adds a full device barrier after
    // each evaluation for debugging or for engines built with private streams.
    if (std::getenv("GMX_WPE_SYNC") != nullptr)
    {
        cudaDeviceSynchronize();
    }
}

} // namespace gmx
