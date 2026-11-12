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
 * \brief Backend selection and engine options for the NEP/qNEP potentials.
 *
 * Kept in a standalone header without CUDA or GPUMD includes so that the
 * option layer can build and validate a backend configuration during grompp
 * on machines without a GPU.
 */
#ifndef GMX_APPLIED_FORCES_NNPOT_NEPBACKENDSETTINGS_H
#define GMX_APPLIED_FORCES_NNPOT_NEPBACKENDSETTINGS_H

#include <string>

namespace gmx
{

/*! \brief Which engine evaluates the NEP/qNEP model. */
enum class NepBackend
{
    //! Use WPE when the model and runtime allow it, otherwise vendored GPUMD.
    Auto,
    //! Vendored GPUMD kernels (CUDA), including GROMACS-owned qNEP electrostatics.
    Gpumd,
    //! Wisevolve Potential Engine (single GPU, proprietary runtime, dlopen).
    Wpe
};

//! Names used in the mdp option and in diagnostics.
const char* nepBackendName(NepBackend backend);
//! Parse "auto", "gpumd" or "wpe" (case insensitive).
NepBackend nepBackendFromString(const std::string& name);

/*! \brief Backend selection and engine options, filled from mdp parameters. */
struct NepBackendSettings
{
    //! Requested backend (nnpot-backend).
    NepBackend backend = NepBackend::Auto;
    //! Explicit WPE shared library / directory (nnpot-wpe-library); empty = search.
    std::string wpeLibrary;
    //! qNEP reciprocal-space method for WPE (pppm|ewald).
    std::string wpeKspace = "pppm";
    //! WPE PPPM target mesh spacing in Angstrom (0.2 - 2.0).
    double wpePppmSpacing = 1.0;
    //! qNEP reciprocal-space method for the native GPUMD backend (pppm|ewald).
    std::string gpumdKspace = "pppm";
    //! Native GPUMD PPPM spacing in Angstrom; <= 0 keeps the model default.
    double gpumdPppmSpacing = 0.0;
    //! Optional D3 correction: "<functional> <rc_potential> <rc_coordination>".
    std::string dftd3;
    //! Ensemble label forwarded to WPE (nve/nvt/npt/...).
    std::string ensembleName = "nve";
    //! Ask the engine for per-atom charges (qNEP charge output).
    bool wantChargeOutput = false;
};

} // namespace gmx

#endif
