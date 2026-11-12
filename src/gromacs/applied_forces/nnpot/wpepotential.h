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
/*! \internal \file
 * \brief Adapter that exposes the Wisevolve Potential Engine as a GPUMD
 * Potential so that the existing NEP device path can use it unchanged.
 */
#ifndef GMX_APPLIED_FORCES_NNPOT_WPEPOTENTIAL_H
#define GMX_APPLIED_FORCES_NNPOT_WPEPOTENTIAL_H

#include <cstdint>
#include <string>

#include "wisevolve_potential_api.h"

#include "force/potential.cuh"
#include "utilities/gpu_vector.cuh"

namespace gmx
{

/*! \brief Static run facts forwarded to wpe_activate_stage().
 *
 * WPE decides its execution policy (including virial routing and whether
 * charge/BEC buffers are needed) from these facts, so they must describe how
 * GROMACS consumes the potential: one live system stage, per-atom virial
 * consumers, and optional charge/BEC outputs.
 */
struct WpeStageSettings
{
    //! PPPM or Ewald for native qNEP electrostatics.
    uint32_t qnepKspaceMethod = 0; // WPE_QNEP_KSPACE_PPPM
    //! PPPM target mesh spacing in Angstrom (0.2 - 2.0, 1.0 default).
    double qnepPppmSpacing = 1.0;
    //! Optional D3 correction (only for NEP-family models that support it).
    bool        dftd3Enabled            = false;
    std::string dftd3Functional;
    float       dftd3RcPotential        = 0.0f;
    float       dftd3RcCoordinationNumber = 0.0f;
    //! Ensemble label (informational for the engine).
    std::string ensembleName = "nve";
    //! Request per-atom charges from the engine (qNEP).
    bool needChargeOutput = false;
    //! Request Born effective charges from the engine.
    bool needBecOutput = false;
    //! Ask for per-atom virials (9N doubles) instead of the cheap system virial.
    //! GROMACS only needs the 6-component system virial for pressure/`ForceWithVirial`,
    //! so the default is the fast path (matches GPUMD's default stage facts).
    bool needAtomVirial = false;
};

/*! \brief GPUMD Potential implementation backed by libwisevolve_potential.so.
 *
 * The engine writes directly into the device buffers that the NEP backend
 * already owns (position/type/force/potential/virial), so no extra staging
 * copies are introduced. WPE uses the CUDA default stream, matching the rest
 * of the external NEP kernels in this module.
 */
class WpePotential final : public ::Potential
{
public:
    WpePotential(const std::string&        potentialPath,
                 int                       numberOfAtoms,
                 const WpeStageSettings&   settings,
                 const std::string&        libraryHint);
    ~WpePotential() override;

    WpePotential(const WpePotential&)            = delete;
    WpePotential& operator=(const WpePotential&) = delete;

    void compute(Box&                      box,
                 const GPU_Vector<int>&    type,
                 const GPU_Vector<double>& position,
                 GPU_Vector<double>&       potential,
                 GPU_Vector<double>&       force,
                 GPU_Vector<double>&       virial) override;

    GPU_Vector<float>& get_charge_reference() override { return charge_; }
    GPU_Vector<float>& get_bec_reference() override { return bec_; }

    //! Re-activate the WPE stage for a new atom count.
    void activateStage(int numberOfAtoms, const WpeStageSettings& settings);

    //! Whether the engine currently writes per-atom charges.
    bool hasChargeOutput() const { return needChargeOutput_; }


    //! Engine metadata for logging (empty before initialization).
    const std::string& engineName() const { return engineName_; }
    const std::string& engineVersion() const { return engineVersion_; }
    const std::string& engineCitation() const { return engineCitation_; }
    const std::string& libraryPath() const { return libraryPath_; }

private:
    WpeContext* context_ = nullptr;

    GPU_Vector<float>  charge_;
    GPU_Vector<float>  bec_;

    bool        needChargeOutput_ = false;
    bool        needBecOutput_    = false;
    bool        needAtomVirial_   = false;
    uint32_t    pimdBeads_        = 0;
    int         atomCount_        = 0;

    std::string libraryPath_;
    std::string engineName_;
    std::string engineVersion_;
    std::string engineCitation_;

    //! Cached int32 PBC flags; the C ABI takes int32_t pointers.
    int32_t pbcX_ = 1;
    int32_t pbcY_ = 1;
    int32_t pbcZ_ = 1;
};

} // namespace gmx

#endif
