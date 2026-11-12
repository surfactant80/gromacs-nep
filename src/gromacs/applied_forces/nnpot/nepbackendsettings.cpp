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

#include "nepbackendsettings.h"

#include <algorithm>
#include <cctype>

#include "gromacs/utility/exceptions.h"

namespace gmx
{

const char* nepBackendName(const NepBackend backend)
{
    switch (backend)
    {
        case NepBackend::Auto: return "auto";
        case NepBackend::Gpumd: return "gpumd";
        case NepBackend::Wpe: return "wpe";
    }
    return "unknown";
}

NepBackend nepBackendFromString(const std::string& name)
{
    std::string lowered(name);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    if (lowered.empty() || lowered == "auto")
    {
        return NepBackend::Auto;
    }
    if (lowered == "gpumd" || lowered == "native")
    {
        return NepBackend::Gpumd;
    }
    if (lowered == "wpe" || lowered == "wisevolve")
    {
        return NepBackend::Wpe;
    }
    throw InconsistentInputError("Unknown NEP backend '" + name
                                 + "' (valid values: auto, gpumd, wpe).");
}

} // namespace gmx
