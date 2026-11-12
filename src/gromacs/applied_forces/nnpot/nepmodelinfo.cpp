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

#include "nepmodelinfo.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringutil.h"

// GPUMD's new-format reader is pure host C++ (no CUDA types in the headers),
// so grompp can use it to validate and inspect new-format models.
#include "utilities/nep_model_file.cuh"

namespace gmx
{

namespace
{

const char* const c_elements[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"
};

std::vector<std::string> firstNonEmptyTokens(const std::string& filename)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw InconsistentInputError("Cannot open NEP model file: " + filename);
    }
    std::string line;
    while (std::getline(input, line))
    {
        if (line.find_first_not_of(" \t\r\n") != std::string::npos)
        {
            std::istringstream stream(line);
            std::vector<std::string> result;
            std::string              token;
            while (stream >> token)
            {
                result.push_back(token);
            }
            return result;
        }
    }
    throw InconsistentInputError("NEP model file is empty: " + filename);
}

bool isQnepLegacyTag(const std::string& tag)
{
    return tag == "nep4_charge1" || tag == "nep4_zbl_charge1" || tag == "nep4_charge2"
           || tag == "nep4_zbl_charge2" || tag == "nep4_charge3" || tag == "nep4_zbl_charge3";
}

int legacyChargeMode(const std::string& tag)
{
    if (tag.find("charge1") != std::string::npos)
    {
        return 1;
    }
    if (tag.find("charge2") != std::string::npos)
    {
        return 2;
    }
    if (tag.find("charge3") != std::string::npos)
    {
        return 3;
    }
    return 0;
}

NepModelFamily familyFromMdTag(const std::string& mdTag)
{
    if (mdTag == "nep_vdw")
    {
        return NepModelFamily::NepVdw;
    }
    if (mdTag == "qnep3_vdw")
    {
        return NepModelFamily::Qnep3Vdw;
    }
    if (mdTag.rfind("nep4_charge", 0) == 0 || mdTag.rfind("nep4_zbl_charge", 0) == 0)
    {
        return NepModelFamily::Qnep;
    }
    if (mdTag == "nep4" || mdTag == "nep4_zbl" || mdTag == "nep4_temperature"
        || mdTag == "nep4_zbl_temperature")
    {
        return NepModelFamily::Nep4;
    }
    return NepModelFamily::Unsupported;
}

[[noreturn]] void throwTruncatedModel(const std::string& filename)
{
    throw InconsistentInputError("NEP model file is malformed or truncated: " + filename);
}

/*! \brief CPU validation of a legacy NEP4/qNEP parameter file.
 *
 * Replicates the arithmetic of the GPUMD parser so a truncated model is
 * rejected by grompp (which has no GPU) instead of aborting mdrun later.
 */
void validateLegacyModel(const std::string&       filename,
                         const std::vector<std::string>& header,
                         int                      numTypes,
                         int                      lMax3body,
                         int                      annHidden,
                         int                      chargeMode,
                         NepModelInfo*            info)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw InconsistentInputError("Cannot open NEP model file: " + filename);
    }
    std::vector<std::string> lines;
    std::string              line;
    while (std::getline(input, line))
    {
        lines.push_back(line);
    }

    size_t lineIndex = 0;
    while (lineIndex < lines.size()
           && lines[lineIndex].find_first_not_of(" \t\r\n") == std::string::npos)
    {
        ++lineIndex;
    }
    const auto tokensAt = [&](size_t index) -> std::vector<std::string>
    {
        if (index >= lines.size())
        {
            throwTruncatedModel(filename);
        }
        std::istringstream stream(lines[index]);
        std::vector<std::string> result;
        std::string              token;
        while (stream >> token)
        {
            result.push_back(token);
        }
        if (result.empty())
        {
            throwTruncatedModel(filename);
        }
        return result;
    };

    ++lineIndex; // header line, already parsed by the caller
    if (info->hasZbl)
    {
        const auto zblTokens = tokensAt(lineIndex++);
        if (zblTokens.size() < 3 || zblTokens.size() > 4 || zblTokens[0] != "zbl")
        {
            throwTruncatedModel(filename);
        }
    }
    const auto cutoffTokens = tokensAt(lineIndex++);
    if (cutoffTokens.size() != 5 || cutoffTokens[0] != "cutoff")
    {
        throwTruncatedModel(filename);
    }
    info->rcRadialA  = std::stod(cutoffTokens[1]);
    info->rcAngularA = std::stod(cutoffTokens[2]);

    const auto nMaxTokens = tokensAt(lineIndex++);
    if (nMaxTokens.size() != 3 || nMaxTokens[0] != "n_max")
    {
        throwTruncatedModel(filename);
    }
    const int nMaxRadial  = std::stoi(nMaxTokens[1]);
    const int nMaxAngular = std::stoi(nMaxTokens[2]);

    const auto basisTokens = tokensAt(lineIndex++);
    if (basisTokens.size() != 3 || basisTokens[0] != "basis_size")
    {
        throwTruncatedModel(filename);
    }
    const int basisRadial  = std::stoi(basisTokens[1]);
    const int basisAngular = std::stoi(basisTokens[2]);

    const auto lMaxTokens = tokensAt(lineIndex++);
    if (lMaxTokens.size() < 4 || lMaxTokens[0] != "l_max")
    {
        throwTruncatedModel(filename);
    }
    int numL = lMax3body;
    for (size_t i = 2; i < lMaxTokens.size(); ++i)
    {
        if (std::stoi(lMaxTokens[i]) != 0)
        {
            ++numL;
        }
    }

    const auto annTokens = tokensAt(lineIndex++);
    if (annTokens.size() != 3 || annTokens[0] != "ANN")
    {
        throwTruncatedModel(filename);
    }

    const int dim        = (nMaxRadial + 1) + (nMaxAngular + 1) * numL;
    const int numTypesSq = numTypes * numTypes;
    // qNEP stores (dim+3) ANN parameters per neuron and type plus two shared
    // values; plain NEP4 stores (dim+2) plus one. Both append dim q_scaler
    // values after the descriptor parameters.
    const int numParaAnn  = chargeMode > 0 ? (dim + 3) * annHidden * numTypes + 2
                                           : (dim + 2) * annHidden * numTypes + 1;
    const int numParaDescr = numTypesSq
                             * ((nMaxRadial + 1) * (basisRadial + 1)
                                + (nMaxAngular + 1) * (basisAngular + 1));
    const size_t expectedValues = static_cast<size_t>(numParaAnn + numParaDescr + dim);

    size_t valueCount = 0;
    for (size_t i = lineIndex; i < lines.size(); ++i)
    {
        std::istringstream stream(lines[i]);
        std::string        token;
        while (stream >> token)
        {
            ++valueCount;
        }
    }
    if (valueCount < expectedValues)
    {
        throw InconsistentInputError(
                "NEP model file is truncated: expected at least "
                + std::to_string(expectedValues) + " parameter values, found "
                + std::to_string(valueCount) + " (" + filename + ").");
    }
}

} // namespace

namespace
{
NepModelInfo readNepModelInfoImpl(const std::string& filename)
{
    const std::vector<std::string> header = firstNonEmptyTokens(filename);
    if (header.size() < 3)
    {
        throw InconsistentInputError("Invalid NEP model header in " + filename + ".");
    }

    NepModelInfo info;
    info.sourceTag = header[0];

    if (info.sourceTag == "nep")
    {
        info.newFormat = true;

        std::ifstream input(filename);
        if (!input)
        {
            throw InconsistentInputError("Cannot open NEP model file: " + filename);
        }
        NEP_Model_File model;
        std::string    error;
        if (!model.read(input, error))
        {
            throw InconsistentInputError(
                    "Cannot parse NEP model file " + filename + ": " + error);
        }

        info.numTypes = model.num_types;
        info.elements = model.elements;
        info.rcRadialA  = model.rc_radial.empty()
                                  ? 0.0
                                  : *std::max_element(model.rc_radial.begin(),
                                                      model.rc_radial.end());
        info.rcAngularA = model.rc_angular.empty()
                                  ? 0.0
                                  : *std::max_element(model.rc_angular.begin(),
                                                      model.rc_angular.end());
        info.lMax3body = model.L_max;
        info.annHidden = model.num_neurons1;
        info.hasZbl    = model.enable_zbl;

        if (model.name == "nep_dipole" || model.name == "nep_polarizability"
            || model.name == "nep_spin")
        {
            info.unsupportedReason =
                    "model family '" + model.name
                    + "' is not an energy/force MD potential for GROMACS";
        }
        else
        {
            std::string mdInput;
            if (!model.get_md_input(mdInput, error))
            {
                info.unsupportedReason = "GPUMD cannot convert this model for MD: " + error;
            }
            else
            {
                std::istringstream stream(mdInput);
                stream >> info.mdTag;
                info.family = familyFromMdTag(info.mdTag);
                if (info.family == NepModelFamily::Unsupported)
                {
                    info.unsupportedReason =
                            "MD tag '" + info.mdTag + "' is not supported by this build";
                }
                info.chargeMode = legacyChargeMode(info.mdTag);
            }
        }
        if (info.elements.size() != static_cast<size_t>(info.numTypes))
        {
            throw InconsistentInputError("Invalid element list in NEP model " + filename + ".");
        }
        return info;
    }

    // Legacy format: the first token is the model tag.
    info.mdTag = info.sourceTag;
    info.numTypes = std::stoi(header[1]);
    if (info.numTypes <= 0 || static_cast<int>(header.size()) != info.numTypes + 2)
    {
        throw InconsistentInputError("Invalid NEP model first line in " + filename + ".");
    }
    info.elements.assign(header.begin() + 2, header.end());

    if (info.mdTag == "nep4" || info.mdTag == "nep4_zbl")
    {
        info.family = NepModelFamily::Nep4;
    }
    else if (isQnepLegacyTag(info.mdTag))
    {
        info.family     = NepModelFamily::Qnep;
        info.chargeMode = legacyChargeMode(info.mdTag);
    }
    else if (info.mdTag == "nep5" || info.mdTag == "nep5_zbl")
    {
        info.unsupportedReason =
                "GPUMD removed NEP5 from the MD potential classes (the current MD backend "
                "implements NEP4 only); NEP5 files are still used by GPUMD's replica-exchange "
                "tool. Retrain as NEP4, or run GPUMD's replica tool for NEP5";
    }
    else if (info.mdTag == "nep4_temperature" || info.mdTag == "nep4_zbl_temperature")
    {
        info.unsupportedReason =
                "temperature-dependent NEP4 models need a temperature-dependent force "
                "evaluation that the GROMACS adapter does not drive";
    }
    else
    {
        info.unsupportedReason = "unknown legacy NEP model tag '" + info.mdTag + "'";
    }
    info.hasZbl = info.mdTag.find("zbl") != std::string::npos;

    // Parse the structural lines needed for cutoffs, WPE shape checks and
    // truncated-file detection. Even unsupported tags are parsed so the error
    // message can be specific.
    std::ifstream input(filename);
    if (!input)
    {
        throw InconsistentInputError("Cannot open NEP model file: " + filename);
    }
    std::vector<std::string> lines;
    std::string              line;
    while (std::getline(input, line))
    {
        lines.push_back(line);
    }
    size_t lineIndex = 0;
    while (lineIndex < lines.size()
           && lines[lineIndex].find_first_not_of(" \t\r\n") == std::string::npos)
    {
        ++lineIndex;
    }
    const auto nextTokens = [&](size_t index) -> std::vector<std::string>
    {
        if (index >= lines.size())
        {
            throwTruncatedModel(filename);
        }
        std::istringstream stream(lines[index]);
        std::vector<std::string> result;
        std::string              token;
        while (stream >> token)
        {
            result.push_back(token);
        }
        if (result.empty())
        {
            throwTruncatedModel(filename);
        }
        return result;
    };
    ++lineIndex;
    if (info.hasZbl)
    {
        ++lineIndex; // zbl line
    }
    // Sanity-check the remaining structure for every tag.
    const auto cutoffTokens = nextTokens(lineIndex++);
    const auto nMaxTokens   = nextTokens(lineIndex++);
    const auto basisTokens  = nextTokens(lineIndex++);
    const auto lMaxTokens   = nextTokens(lineIndex++);
    const auto annTokens    = nextTokens(lineIndex++);
    if (cutoffTokens[0] != "cutoff" || cutoffTokens.size() != 5 || nMaxTokens[0] != "n_max"
        || nMaxTokens.size() != 3 || basisTokens[0] != "basis_size"
        || basisTokens.size() != 3 || lMaxTokens[0] != "l_max" || lMaxTokens.size() < 4
        || annTokens[0] != "ANN" || annTokens.size() != 3)
    {
        throwTruncatedModel(filename);
    }
    info.rcRadialA  = std::stod(cutoffTokens[1]);
    info.rcAngularA = std::stod(cutoffTokens[2]);
    info.lMax3body  = std::stoi(lMaxTokens[1]);
    info.annHidden  = std::stoi(annTokens[1]);

    if (info.family != NepModelFamily::Unsupported)
    {
        validateLegacyModel(
                filename, header, info.numTypes, info.lMax3body, info.annHidden, info.chargeMode, &info);
    }
    return info;
}

} // namespace

bool isQnepModel(const NepModelInfo& info)
{
    return info.family == NepModelFamily::Qnep || info.family == NepModelFamily::Qnep3Vdw;
}

bool gpumdSupportsModel(const NepModelInfo& info, std::string* reason)
{
    if (info.family == NepModelFamily::Unsupported)
    {
        if (reason != nullptr)
        {
            *reason = info.unsupportedReason;
        }
        return false;
    }
#if !GMX_NEP_HAVE_VDW
    if (info.family == NepModelFamily::NepVdw || info.family == NepModelFamily::Qnep3Vdw)
    {
        if (reason != nullptr)
        {
            *reason = "this GROMACS build was linked against a GPUMD tree without "
                      "NEP_Charge_VDW (vdW NEP); rebuild with a GPUMD master that "
                      "provides force/nep_charge_vdw/";
        }
        return false;
    }
#endif
    return true;
}

bool wpeSupportsModel(const NepModelInfo& info, std::string* reason)
{
    if (info.family == NepModelFamily::NepVdw || info.family == NepModelFamily::Qnep3Vdw)
    {
        if (reason != nullptr)
        {
            *reason = "the WPE 1.0.0 release supports NEP4/NEP4-ZBL, qNEP, LJ, EAM and "
                      "Tersoff, but not the vdW NEP families";
        }
        return false;
    }
    if (info.family == NepModelFamily::Unsupported)
    {
        if (reason != nullptr)
        {
            *reason = info.unsupportedReason;
        }
        return false;
    }
    if (info.lMax3body != 4)
    {
        if (reason != nullptr)
        {
            *reason = formatString(
                    "WPE requires l_max_3body = 4 (model has %d). Use nnpot-backend = gpumd.",
                    info.lMax3body);
        }
        return false;
    }
    if (info.annHidden > 256)
    {
        if (reason != nullptr)
        {
            *reason = formatString(
                    "WPE requires an ANN hidden width <= 256 (model has %d). Use "
                    "nnpot-backend = gpumd.",
                    info.annHidden);
        }
        return false;
    }
    return true;
}

NepModelInfo readNepModelInfo(const std::string& filename)
{
    try
    {
        return readNepModelInfoImpl(filename);
    }
    catch (const GromacsException&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw InconsistentInputError("Cannot parse NEP model file '" + filename
                                     + "': " + error.what());
    }
}

std::string describeNepModel(const NepModelInfo& info)
{
    const char* family = "unknown";
    switch (info.family)
    {
        case NepModelFamily::Nep4: family = "NEP4"; break;
        case NepModelFamily::Qnep: family = "qNEP"; break;
        case NepModelFamily::NepVdw: family = "NEP-vdW"; break;
        case NepModelFamily::Qnep3Vdw: family = "qNEP3-vdW"; break;
        case NepModelFamily::Unsupported: family = "unsupported"; break;
    }
    return formatString("%s (%s, tag %s, %d type%s, r_c = %.3f / %.3f A, l_max_3body = %d, "
                        "ANN hidden = %d)",
                        family,
                        info.newFormat ? "new format" : "legacy format",
                        info.mdTag.c_str(),
                        info.numTypes,
                        info.numTypes == 1 ? "" : "s",
                        info.rcRadialA,
                        info.rcAngularA,
                        info.lMax3body,
                        info.annHidden);
}

} // namespace gmx
