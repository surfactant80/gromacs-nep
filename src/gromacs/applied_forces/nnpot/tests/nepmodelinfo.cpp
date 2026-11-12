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

#include "gromacs/applied_forces/nnpot/nepmodelinfo.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "gromacs/utility/exceptions.h"

#include "testutils/testfilemanager.h"

namespace gmx
{
namespace
{

/*! \brief Write a syntactically complete legacy NEP4 model and return its path. */
std::string writeLegacyModel(gmx::test::TestFileManager* fileManager,
                             const std::string&          name,
                             const int                   lMax3body,
                             const int                   annHidden,
                             const bool                  truncated)
{
    const int nMaxRadial  = 2;
    const int nMaxAngular = 2;
    const int basis       = 4;
    const int numL        = lMax3body; // no higher-order q flags
    const int dim         = (nMaxRadial + 1) + (nMaxAngular + 1) * numL;
    const int numParaAnn  = (dim + 2) * annHidden + 1;
    const int numDescr    = (nMaxRadial + 1) * (basis + 1) + (nMaxAngular + 1) * (basis + 1);
    const int numValues   = numParaAnn + numDescr + dim;

    const std::string path = fileManager->getTemporaryFilePath(name);
    std::ofstream     output(path);
    output << "nep4 1 C\n";
    output << "cutoff 8.0 4.0 100 100\n";
    output << "n_max " << nMaxRadial << " " << nMaxAngular << "\n";
    output << "basis_size " << basis << " " << basis << "\n";
    output << "l_max " << lMax3body << " 0 0\n";
    output << "ANN " << annHidden << " 0\n";
    const int values = truncated ? numValues - 10 : numValues;
    for (int i = 0; i < values; ++i)
    {
        output << "0.0\n";
    }
    output.close();
    return path;
}

TEST(NepModelInfoTest, ParsesLegacyNep4Model)
{
    gmx::test::TestFileManager fileManager;
    const std::string path = writeLegacyModel(&fileManager, "nep4-ok.txt", 4, 30, false);
    const NepModelInfo info = readNepModelInfo(path);
    EXPECT_FALSE(info.newFormat);
    EXPECT_EQ(info.family, NepModelFamily::Nep4);
    EXPECT_EQ(info.numTypes, 1);
    EXPECT_EQ(info.elements.at(0), "C");
    EXPECT_DOUBLE_EQ(info.rcRadialA, 8.0);
    EXPECT_DOUBLE_EQ(info.rcAngularA, 4.0);
    EXPECT_EQ(info.lMax3body, 4);
    EXPECT_EQ(info.annHidden, 30);
    EXPECT_TRUE(isQnepModel(info) == false);
    std::string reason;
    EXPECT_TRUE(gpumdSupportsModel(info, &reason));
    EXPECT_TRUE(wpeSupportsModel(info, &reason)) << reason;
}

TEST(NepModelInfoTest, RejectsTruncatedModel)
{
    gmx::test::TestFileManager fileManager;
    const std::string path = writeLegacyModel(&fileManager, "nep4-bad.txt", 4, 30, true);
    EXPECT_THROW(readNepModelInfo(path), InconsistentInputError);
}

TEST(NepModelInfoTest, WpeRequiresLmax3body4)
{
    gmx::test::TestFileManager fileManager;
    const std::string path = writeLegacyModel(&fileManager, "nep4-lmax3.txt", 3, 30, false);
    const NepModelInfo info = readNepModelInfo(path);
    std::string reason;
    EXPECT_FALSE(wpeSupportsModel(info, &reason));
    EXPECT_NE(reason.find("l_max_3body"), std::string::npos);
    EXPECT_TRUE(gpumdSupportsModel(info, &reason));
}

TEST(NepModelInfoTest, WpeRequiresHiddenWidthAtMost256)
{
    gmx::test::TestFileManager fileManager;
    const std::string path = writeLegacyModel(&fileManager, "nep4-wide.txt", 4, 300, false);
    NepModelInfo       info = readNepModelInfo(path);
    // The legacy parameter-count check is independent of the ANN width; the
    // WPE shape limit is a separate, later check.
    EXPECT_EQ(info.annHidden, 300);
    std::string reason;
    EXPECT_FALSE(wpeSupportsModel(info, &reason));
    EXPECT_NE(reason.find("256"), std::string::npos);
}

TEST(NepModelInfoTest, UnsupportedLegacyTagIsReported)
{
    gmx::test::TestFileManager fileManager;
    const std::string path = fileManager.getTemporaryFilePath("nep5.txt");
    std::ofstream     output(path);
    output << "nep5 1 C\ncutoff 8.0 4.0 100 100\nn_max 2 2\nbasis_size 4 4\n"
              "l_max 4 0 0\nANN 30 0\n";
    output.close();
    const NepModelInfo info = readNepModelInfo(path);
    EXPECT_EQ(info.family, NepModelFamily::Unsupported);
    EXPECT_FALSE(info.unsupportedReason.empty());
    std::string reason;
    EXPECT_FALSE(gpumdSupportsModel(info, &reason));
}

} // namespace
} // namespace gmx
