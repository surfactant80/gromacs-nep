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
 * \brief CPU-side NEP/qNEP model inspection shared by grompp and mdrun.
 *
 * Parsing here never touches CUDA: grompp can validate a model without a GPU
 * and without loading the proprietary WPE runtime. Both the legacy
 * (first-line tag, e.g. "nep4 6 H C N O S P") and the new YAML-like
 * "nep <N> <elements>" format are supported by delegating the new format to
 * GPUMD's pure-C++ NEP_Model_File reader.
 */
#ifndef GMX_APPLIED_FORCES_NNPOT_NEPMODELINFO_H
#define GMX_APPLIED_FORCES_NNPOT_NEPMODELINFO_H

#include <string>
#include <vector>

namespace gmx
{

//! Potential family a model belongs to.
enum class NepModelFamily
{
    //! Plain NEP4 (optionally ZBL, optionally temperature dependent).
    Nep4,
    //! qNEP with dynamic charges (charge1/charge2/charge3).
    Qnep,
    //! vdW-aware NEP (nep_vdw).
    NepVdw,
    //! vdW-aware qNEP (qnep3_vdw).
    Qnep3Vdw,
    //! Recognized but not usable as a GROMACS MD potential.
    Unsupported
};

/*! \brief Metadata extracted from a NEP model file on the CPU. */
struct NepModelInfo
{
    //! True for the new "nep <N> <elements>" format.
    bool newFormat = false;
    //! Tag exactly as it appears in the file ("nep4", "qnep1", ...).
    std::string sourceTag;
    //! Tag that the GPUMD MD readers expect after format conversion.
    std::string mdTag;
    //! Potential family.
    NepModelFamily family = NepModelFamily::Unsupported;
    //! Number of atom types.
    int numTypes = 0;
    //! Element symbols in model type order.
    std::vector<std::string> elements;
    //! Radial cutoff in Angstrom.
    double rcRadialA = 0.0;
    //! Angular cutoff in Angstrom.
    double rcAngularA = 0.0;
    //! L_max of the 3-body descriptor (WPE requires exactly 4).
    int lMax3body = 0;
    //! Width of the ANN hidden layer (WPE requires <= 256).
    int annHidden = 0;
    //! Universal ZBL is enabled.
    bool hasZbl = false;
    //! qNEP charge mode (0 for plain NEP).
    int chargeMode = 0;
    //! Reason the model cannot be used, when family == Unsupported.
    std::string unsupportedReason;
};

/*! \brief Parse and validate a NEP model file without initializing CUDA.
 *
 * Throws gmx::InconsistentInputError for malformed or truncated files and for
 * model families that the GROMACS MD interface cannot evaluate.
 */
NepModelInfo readNepModelInfo(const std::string& filename);

//! Whether the model is a qNEP (dynamic charge) model.
bool isQnepModel(const NepModelInfo& info);

//! Whether the vendored GPUMD backend can evaluate the model.
bool gpumdSupportsModel(const NepModelInfo& info, std::string* reason);

/*! \brief Whether WPE can be expected to load the model.
 *
 * Only static model properties are checked here (family and shape limits from
 * the WPE release notes). The engine remains the authority: wpe_create() may
 * still reject a model and then reports WPE_ERR_UNSUPPORTED_MODEL.
 */
bool wpeSupportsModel(const NepModelInfo& info, std::string* reason);

//! One-line human readable model description for log output.
std::string describeNepModel(const NepModelInfo& info);

} // namespace gmx

#endif
