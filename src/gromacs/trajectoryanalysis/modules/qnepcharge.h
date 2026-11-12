/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */
/*! \internal \file
 * \brief
 * Declares the gmx qnep-charge trajectory analysis module.
 */
#ifndef GMX_TRAJECTORYANALYSIS_MODULES_CHARGE_H
#define GMX_TRAJECTORYANALYSIS_MODULES_CHARGE_H

#include "gromacs/trajectoryanalysis/analysismodule.h"

namespace gmx::analysismodules
{

class QnepChargeInfo
{
public:
    static const char                      name[];
    static const char                      shortDescription[];
    static TrajectoryAnalysisModulePointer create();
};

} // namespace gmx::analysismodules

#endif // GMX_TRAJECTORYANALYSIS_MODULES_CHARGE_H
