/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Implements the options for NNPot MDModule class.
 *
 * \author Lukas Müllender <lukas.muellender@gmail.com>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "nnpotoptions.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/atomprop.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/stringutil.h"

#include "nnpot.h"

#if GMX_NEP
#    include "gromacs/applied_forces/nnpot/nepmodel.h"
#endif
#if GMX_TORCH
#    include "gromacs/applied_forces/nnpot/torchmodel.h"
#endif

namespace gmx
{

namespace
{

//! Helper function to make a std::string containing the module name
std::string moduleName()
{
    return std::string(NNPotModuleInfo::sc_name);
}

/*! \brief Following Tags denote names of parameters from .mdp file
 * \note Changing these strings will break .tpr backwards compability
 */
//! \{
const std::string c_activeTag_        = "active";
const std::string c_modelFileNameTag_ = "modelfile";
const std::string c_inputGroupTag_    = "input-group";
const std::string c_linkTypeTag_      = "link-type";
const std::string c_linkDistanceTag_  = "link-distance";
const std::string c_pairCutoffTag_    = "pair-cutoff";
const std::string c_nnpChargeTag_     = "nnp-charge";
const std::string c_embeddingTag_     = "embedding";
const std::string c_useGromacsElectrostaticsTag_ = "use-gromacs-electrostatics";
const std::string c_gromacsChargesTag_ = "gromacs-charges";
const std::string c_energyOffsetsTag_ = "energy-offsets";
const std::string c_adaptiveModeTag_ = "adaptive-mode";
const std::string c_adaptiveGroupBTag_ = "adaptive-group-b";
const std::string c_adaptiveModelFileNameTag_ = "adaptive-modelfile";
const std::string c_adaptiveModelIsDeltaTag_ = "adaptive-model-is-delta";
const std::string c_adaptiveInnerCutoffTag_ = "adaptive-inner-cutoff";
const std::string c_adaptiveOuterCutoffTag_ = "adaptive-outer-cutoff";

//! The names of the supported embedding schemes
const EnumerationArray<NNPotEmbedding, const char*> c_embeddingSchemeNames = {
    { "mechanical", "electrostatic-model" }
};
const EnumerationArray<NNPotAdaptiveMode, const char*> c_adaptiveModeNames = {
    { "off", "cross-only", "a-nep-cross" }
};

/*! \brief User defined input to NN model.
 *
 *  Possible values:
 * - "atom-positions" vector of atom positions
 * - "atom-numbers" vector of atom numbers
 * - "atom-pairs" list of atom pairs within cutoff set by user
 * - "pair-shifts" list of periodic shifts for atom pairs
 * - "box" unit vectors of simulation box
 * - "pbc" boolean vector indicating periodic boundary conditions
 * - "atom-positions-mm" vector of atom positions
 * - "atom-charges-mm" vector of atomic charges
 * - "nnp-charge" charge of the NNP region
 */
const std::string c_modelInput1Tag_ = "model-input1";
const std::string c_modelInput2Tag_ = "model-input2";
const std::string c_modelInput3Tag_ = "model-input3";
const std::string c_modelInput4Tag_ = "model-input4";
const std::string c_modelInput5Tag_ = "model-input5";
const std::string c_modelInput6Tag_ = "model-input6";
const std::string c_modelInput7Tag_ = "model-input7";
const std::string c_modelInput8Tag_ = "model-input8";
const std::string c_modelInput9Tag_ = "model-input9";
//! \}

/*! \brief Following tags are needed to write parameters generated
 * during preprocessing (grompp) to the .tpr file via KVT
 */
//! \{
const std::string c_mmGroupTag_ = "mm-group";
const std::string c_nnpLinkTag_ = "nnp-link";
const std::string c_mmLinkTag_  = "mm-link";
const std::string c_adaptiveBGroupTag_ = "adaptive-b-group";
const std::string c_adaptiveBMoleculeStartsTag_ = "adaptive-b-molecule-starts";
const std::string c_adaptiveBMoleculeSizesTag_ = "adaptive-b-molecule-sizes";
//! \}

std::string readModelHeader(const std::string& filename)
{
    std::ifstream input(filename);
    std::string header;
    input >> header;
    return header;
}

bool isQnepHeader(const std::string& header)
{
    return header == "nep4_charge1" || header == "nep4_zbl_charge1"
           || header == "nep4_charge2" || header == "nep4_zbl_charge2";
}

void disableFullSystemClassicalNonbonded(gmx_mtop_t* mtop, const MDLogger& logger)
{
    // A global intermolecular exclusion group containing every NNP atom is
    // equivalent to an O(N^2) exclusion list. It makes GROMACS neighbor
    // searching dominate full-system ML simulations. When every atom belongs
    // to the NNP region, zeroing all classical charges and atom-type pair
    // parameters is physically equivalent and preserves an O(N) pair search.
    for (auto& moleculeType : mtop->moltype)
    {
        for (int atomIndex = 0; atomIndex < moleculeType.atoms.nr; ++atomIndex)
        {
            auto& atom = moleculeType.atoms.atom[atomIndex];
            atom.q  = 0;
            atom.qB = 0;
        }
    }

    const int numNonbondedParameters = mtop->ffparams.atnr * mtop->ffparams.atnr;
    GMX_RELEASE_ASSERT(
            static_cast<int>(mtop->ffparams.iparams.size()) >= numNonbondedParameters,
            "The force-field parameter table is smaller than the atom-type pair matrix.");
    for (int parameterIndex = 0; parameterIndex < numNonbondedParameters; ++parameterIndex)
    {
        switch (mtop->ffparams.functype[parameterIndex])
        {
            case InteractionFunction::LennardJonesShortRange:
                mtop->ffparams.iparams[parameterIndex].lj.c6  = 0;
                mtop->ffparams.iparams[parameterIndex].lj.c12 = 0;
                break;
            case InteractionFunction::BuckinghamShortRange:
                mtop->ffparams.iparams[parameterIndex].bham.a = 0;
                mtop->ffparams.iparams[parameterIndex].bham.b = 0;
                mtop->ffparams.iparams[parameterIndex].bham.c = 0;
                break;
            default:
                GMX_RELEASE_ASSERT(false,
                                   "Unsupported classical non-bonded function for a "
                                   "full-system neural-network potential.");
        }
    }

    GMX_LOG(logger.info)
            .appendText("Full-system NNP detected: classical charges and non-bonded "
                        "pair parameters were disabled without constructing a global "
                        "intermolecular exclusion group.");
}

/*! \brief Helper function to preprocess topology for NNP
 *
 * This function performs the following modifications:
 * - Excludes non-bonded interactions between NNP atoms (LJ and Coulomb)
 * - In case of electrostatic embedding: removes classical charges on NNP atoms
 * - Removes bonds containing 1 or more NNP atoms
 * - Removes angles and settles containing 2 or more NNP atoms
 * - Removes dihedrals containing 3 or more NNP atoms
 */
std::vector<LinkFrontierAtom> preprocessNNPotTopology(gmx_mtop_t*           mtop,
                                                      ArrayRef<const Index> nnpIndices,
                                                      const NNPotEmbedding& embedding,
                                                      const real&           nnpCharge,
                                                      const bool            keepClassicalElectrostatics,
                                                      const MDLogger&       logger,
                                                      WarningHandler*       wi)
{
    // convert nnpIndices to set for faster lookup
    std::set<int> nnpIndicesSet(nnpIndices.begin(), nnpIndices.end());
    const int     numNNPAtoms     = nnpIndices.size();
    const int     numRegularAtoms = mtop->natoms - numNNPAtoms;

    GMX_LOG(logger.info)
            .appendText("Neural network potential interface is active, topology was modified!");
    GMX_LOG(logger.info)
            .appendTextFormatted("Number of embedded NNP atoms: %d\nNumber of regular atoms: %d\n",
                                 numNNPAtoms,
                                 numRegularAtoms);

    // 1) Split QM-containing molecules from other molecules in blocks
    std::vector<bool> isNNPBlock = splitEmbeddedBlocks(mtop, nnpIndicesSet);

    // 2) Exclude classical non-bonded interactions between NNP atoms.
    // A full-system NNP can disable them directly and avoid a quadratic
    // intermolecular exclusion group.
    if (numNNPAtoms == mtop->natoms && !keepClassicalElectrostatics)
    {
        disableFullSystemClassicalNonbonded(mtop, logger);
    }
    else if (numNNPAtoms == mtop->natoms)
    {
        GMX_LOG(logger.info)
                .appendText("Full-system residual NEP: preserving topology charges and "
                            "GROMACS electrostatics/PME.");
        const int numNonbondedParameters = mtop->ffparams.atnr * mtop->ffparams.atnr;
        for (int parameterIndex = 0; parameterIndex < numNonbondedParameters; ++parameterIndex)
        {
            switch (mtop->ffparams.functype[parameterIndex])
            {
                case InteractionFunction::LennardJonesShortRange:
                    mtop->ffparams.iparams[parameterIndex].lj.c6  = 0;
                    mtop->ffparams.iparams[parameterIndex].lj.c12 = 0;
                    break;
                case InteractionFunction::BuckinghamShortRange:
                    mtop->ffparams.iparams[parameterIndex].bham.a = 0;
                    mtop->ffparams.iparams[parameterIndex].bham.b = 0;
                    mtop->ffparams.iparams[parameterIndex].bham.c = 0;
                    break;
                default:
                    GMX_RELEASE_ASSERT(false,
                                       "Unsupported classical non-bonded function for a "
                                       "full-system residual NEP.");
            }
        }
    }
    else
    {
        addEmbeddedNBExclusions(mtop, nnpIndicesSet, logger);
    }

    // 3) Remove classical charges from embedded atoms if electrostatic embedding is used
    if (embedding == NNPotEmbedding::ElectrostaticModel)
    {
        GMX_LOG(logger.info).appendText("Electrostatic embedding scheme is used.\n");
        removeEmbeddedClassicalCharges(mtop, nnpIndicesSet, isNNPBlock, nnpCharge, logger, wi);
    }

    // 4) Make F_CONNBOND between atoms within QM region
    modifyEmbeddedTwoCenterInteractions(mtop, nnpIndicesSet, isNNPBlock, logger);

    // 5) Remove angles and settles containing 2 or more QM atoms
    modifyEmbeddedThreeCenterInteractions(mtop, nnpIndicesSet, isNNPBlock, logger);

    // 6) Remove dihedrals containing 3 or more QM atoms
    modifyEmbeddedFourCenterInteractions(mtop, nnpIndicesSet, isNNPBlock, logger);

    // 7) Check for constrained bonds in subsystem
    checkConstrainedBonds(mtop, nnpIndicesSet, isNNPBlock, wi);

    // 8) Build link frontier information
    std::vector<LinkFrontierAtom> linkFrontier =
            buildLinkFrontier(mtop, nnpIndicesSet, isNNPBlock, logger);

    // finalize topology
    mtop->finalize();

    return linkFrontier;
}

/*! \brief Clears all exclusions and scaled 1-4 interactions for the qNEP
 * GROMACS-owned electrostatics mode.
 *
 * Native GPUMD qNEP has no topology concept, so every atom pair must interact
 * through the GROMACS Coulomb/PME calculation. The dynamic charges are
 * predicted by qNEP and uploaded every step; the classical charges left in the
 * topology are only placeholders. Classical LJ is disabled separately.
 */
void clearExclusionsForQnepGromacs(gmx_mtop_t* mtop)
{
    const std::array<InteractionFunction, 3> fourteenTypes = {
        InteractionFunction::LennardJones14,
        InteractionFunction::Coulomb14,
        InteractionFunction::LennardJonesCoulomb14Q,
    };
    for (auto& moleculeType : mtop->moltype)
    {
        gmx::ListOfLists<int> emptyExclusions;
        for (int atom = 0; atom < moleculeType.atoms.nr; ++atom)
        {
            emptyExclusions.pushBack(gmx::ArrayRef<const int>());
        }
        moleculeType.excls = std::move(emptyExclusions);
        for (const InteractionFunction ftype : fourteenTypes)
        {
            moleculeType.ilist[ftype].iatoms.clear();
        }
    }
}

} // namespace

void NNPotOptions::initMdpTransform(IKeyValueTreeTransformRules* rules)
{
    const auto& stringIdentityTransform = [](std::string s) { return s; };
    addMdpTransformFromString<bool>(rules, &fromStdString<bool>, NNPotModuleInfo::sc_name, c_activeTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelFileNameTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_inputGroupTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_linkTypeTag_);
    addMdpTransformFromString<real>(
            rules, &fromStdString<real>, NNPotModuleInfo::sc_name, c_linkDistanceTag_);
    addMdpTransformFromString<real>(rules, &fromStdString<real>, NNPotModuleInfo::sc_name, c_pairCutoffTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_embeddingTag_);
    addMdpTransformFromString<bool>(rules,
                                    &fromStdString<bool>,
                                    NNPotModuleInfo::sc_name,
                                    c_useGromacsElectrostaticsTag_);
    addMdpTransformFromString<bool>(
            rules, &fromStdString<bool>, NNPotModuleInfo::sc_name, c_gromacsChargesTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_energyOffsetsTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_adaptiveModeTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_adaptiveGroupBTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_adaptiveModelFileNameTag_);
    addMdpTransformFromString<bool>(rules,
                                    &fromStdString<bool>,
                                    NNPotModuleInfo::sc_name,
                                    c_adaptiveModelIsDeltaTag_);
    addMdpTransformFromString<real>(
            rules, &fromStdString<real>, NNPotModuleInfo::sc_name, c_adaptiveInnerCutoffTag_);
    addMdpTransformFromString<real>(
            rules, &fromStdString<real>, NNPotModuleInfo::sc_name, c_adaptiveOuterCutoffTag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput1Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput2Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput3Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput4Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput5Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput6Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput7Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput8Tag_);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, NNPotModuleInfo::sc_name, c_modelInput9Tag_);
}

void NNPotOptions::initMdpOptions(IOptionsContainerWithSections* options)
{
    auto section = options->addSection(OptionSection(moduleName().c_str()));
    section.addOption(BooleanOption(c_activeTag_.c_str()).store(&params_.active_));
    section.addOption(StringOption(c_modelFileNameTag_.c_str()).store(&params_.modelFileName_));
    section.addOption(StringOption(c_inputGroupTag_.c_str()).store(&params_.inputGroup_));
    section.addOption(StringOption(c_linkTypeTag_.c_str()).store(&params_.linkType_));
    section.addOption(RealOption(c_linkDistanceTag_.c_str()).store(&params_.linkDistance_));
    section.addOption(RealOption(c_pairCutoffTag_.c_str()).store(&params_.pairCutoff_));
    section.addOption(EnumOption<NNPotEmbedding>(c_embeddingTag_.c_str())
                              .enumValue(c_embeddingSchemeNames)
                              .store(&params_.embeddingScheme_));
    section.addOption(BooleanOption(c_useGromacsElectrostaticsTag_.c_str())
                              .store(&params_.useGromacsElectrostatics_));
    section.addOption(
            BooleanOption(c_gromacsChargesTag_.c_str()).store(&params_.gromacsCharges_));
    section.addOption(StringOption(c_energyOffsetsTag_.c_str()).store(&params_.energyOffsets_));
    section.addOption(EnumOption<NNPotAdaptiveMode>(c_adaptiveModeTag_.c_str())
                              .enumValue(c_adaptiveModeNames)
                              .store(&params_.adaptiveMode_));
    section.addOption(StringOption(c_adaptiveGroupBTag_.c_str()).store(&params_.adaptiveGroupB_));
    section.addOption(
            StringOption(c_adaptiveModelFileNameTag_.c_str()).store(&params_.adaptiveModelFileName_));
    section.addOption(BooleanOption(c_adaptiveModelIsDeltaTag_.c_str())
                              .store(&params_.adaptiveModelIsDelta_));
    section.addOption(
            RealOption(c_adaptiveInnerCutoffTag_.c_str()).store(&params_.adaptiveInnerCutoff_));
    section.addOption(
            RealOption(c_adaptiveOuterCutoffTag_.c_str()).store(&params_.adaptiveOuterCutoff_));
    section.addOption(RealOption(c_nnpChargeTag_.c_str()).store(&params_.nnpCharge_));
    section.addOption(StringOption(c_modelInput1Tag_.c_str()).store(&params_.modelInput_[0]));
    section.addOption(StringOption(c_modelInput2Tag_.c_str()).store(&params_.modelInput_[1]));
    section.addOption(StringOption(c_modelInput3Tag_.c_str()).store(&params_.modelInput_[2]));
    section.addOption(StringOption(c_modelInput4Tag_.c_str()).store(&params_.modelInput_[3]));
    section.addOption(StringOption(c_modelInput5Tag_.c_str()).store(&params_.modelInput_[4]));
    section.addOption(StringOption(c_modelInput6Tag_.c_str()).store(&params_.modelInput_[5]));
    section.addOption(StringOption(c_modelInput7Tag_.c_str()).store(&params_.modelInput_[6]));
    section.addOption(StringOption(c_modelInput8Tag_.c_str()).store(&params_.modelInput_[7]));
    section.addOption(StringOption(c_modelInput9Tag_.c_str()).store(&params_.modelInput_[8]));
}

void NNPotOptions::buildMdpOutput(KeyValueTreeObjectBuilder* builder) const
{
    // new empty line before writing nnpot mdp values
    addMdpOutputComment(builder, NNPotModuleInfo::sc_name, "empty-line", "");
    addMdpOutputComment(builder, NNPotModuleInfo::sc_name, "module", "; Neural Network potential");
    addMdpOutputValue(builder, NNPotModuleInfo::sc_name, c_activeTag_, params_.active_);

    if (params_.active_)
    {
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelFileNameTag_, params_.modelFileName_);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_inputGroupTag_, params_.inputGroup_);
        addMdpOutputValue<std::string>(builder, NNPotModuleInfo::sc_name, c_linkTypeTag_, params_.linkType_);
        addMdpOutputValue<real>(builder, NNPotModuleInfo::sc_name, c_linkDistanceTag_, params_.linkDistance_);
        addMdpOutputValue<real>(builder, NNPotModuleInfo::sc_name, c_pairCutoffTag_, params_.pairCutoff_);
        addMdpOutputValue<std::string>(builder,
                                       NNPotModuleInfo::sc_name,
                                       c_embeddingTag_,
                                       c_embeddingSchemeNames[params_.embeddingScheme_]);
        addMdpOutputValue<bool>(builder,
                                NNPotModuleInfo::sc_name,
                                c_useGromacsElectrostaticsTag_,
                                params_.useGromacsElectrostatics_);
        addMdpOutputValue<bool>(builder,
                                NNPotModuleInfo::sc_name,
                                c_gromacsChargesTag_,
                                params_.gromacsCharges_);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_energyOffsetsTag_, params_.energyOffsets_);
        addMdpOutputValue<std::string>(builder,
                                       NNPotModuleInfo::sc_name,
                                       c_adaptiveModeTag_,
                                       c_adaptiveModeNames[params_.adaptiveMode_]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_adaptiveGroupBTag_, params_.adaptiveGroupB_);
        addMdpOutputValue<std::string>(builder,
                                       NNPotModuleInfo::sc_name,
                                       c_adaptiveModelFileNameTag_,
                                       params_.adaptiveModelFileName_);
        addMdpOutputValue<bool>(builder,
                                NNPotModuleInfo::sc_name,
                                c_adaptiveModelIsDeltaTag_,
                                params_.adaptiveModelIsDelta_);
        addMdpOutputValue<real>(builder,
                                NNPotModuleInfo::sc_name,
                                c_adaptiveInnerCutoffTag_,
                                params_.adaptiveInnerCutoff_);
        addMdpOutputValue<real>(builder,
                                NNPotModuleInfo::sc_name,
                                c_adaptiveOuterCutoffTag_,
                                params_.adaptiveOuterCutoff_);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput1Tag_, params_.modelInput_[0]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput2Tag_, params_.modelInput_[1]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput3Tag_, params_.modelInput_[2]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput4Tag_, params_.modelInput_[3]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput5Tag_, params_.modelInput_[4]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput6Tag_, params_.modelInput_[5]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput7Tag_, params_.modelInput_[6]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput8Tag_, params_.modelInput_[7]);
        addMdpOutputValue<std::string>(
                builder, NNPotModuleInfo::sc_name, c_modelInput9Tag_, params_.modelInput_[8]);
    }
}

bool NNPotOptions::isActive() const
{
    return params_.active_;
}

std::string NNPotOptions::getModelFileName() const
{
    return params_.modelFileName_;
}

void NNPotOptions::setInputGroupIndices(const IndexGroupsAndNames& indexGroupsAndNames)
{
    // Exit if NNPot module is not active
    if (!params_.active_)
    {
        return;
    }

#if GMX_NEP
    // NEP always consumes Cartesian positions. Keep the existing model-input
    // mechanism for Torch models, but provide the one required input
    // automatically for a GPUMD NEP text model.
    const std::filesystem::path modelPath(params_.modelFileName_);
    if (modelPath.extension() == ".txt" || modelPath.extension() == ".nep")
    {
        params_.modelInput_[0] = "atom-positions";
    }
#endif

    // Create input index
    params_.nnpIndices_ = indexGroupsAndNames.indices(params_.inputGroup_);

    // Check that group is not empty
    if (params_.nnpIndices_.empty())
    {
        GMX_THROW(InconsistentInputError(
                formatString("Group %s defining NN potential input atoms should not be empty.",
                             params_.inputGroup_.c_str())));
    }
    std::sort(params_.nnpIndices_.begin(), params_.nnpIndices_.end());

    if (params_.adaptiveMode_ != NNPotAdaptiveMode::Off)
    {
        if (params_.adaptiveGroupB_.empty())
        {
            GMX_THROW(InconsistentInputError(
                    "Adaptive NEP/MM requires nnpot-adaptive-group-b."));
        }
        params_.adaptiveBIndices_ = indexGroupsAndNames.indices(params_.adaptiveGroupB_);
        if (params_.adaptiveBIndices_.empty())
        {
            GMX_THROW(InconsistentInputError(
                    "The adaptive B group must not be empty."));
        }
        std::sort(params_.adaptiveBIndices_.begin(), params_.adaptiveBIndices_.end());
        if (std::any_of(params_.adaptiveBIndices_.begin(),
                        params_.adaptiveBIndices_.end(),
                        [this](const Index index)
                        { return std::binary_search(params_.nnpIndices_.begin(),
                                                    params_.nnpIndices_.end(),
                                                    index); }))
        {
            GMX_THROW(InconsistentInputError(
                    "Adaptive A and B groups must be disjoint."));
        }
        if (params_.adaptiveModelFileName_.empty())
        {
            params_.adaptiveModelFileName_ = params_.modelFileName_;
        }
        if (!params_.adaptiveModelIsDelta_)
        {
            GMX_THROW(InconsistentInputError(
                    "Adaptive NEP/MM requires nnpot-adaptive-model-is-delta = yes. "
                    "The adaptive model must be trained on NEP-minus-MM residual labels; "
                    "a total-energy NEP would double count the MM A-B interaction."));
        }
        if (params_.adaptiveOuterCutoff_ <= 0
            || params_.adaptiveInnerCutoff_ < 0
            || params_.adaptiveInnerCutoff_ > params_.adaptiveOuterCutoff_)
        {
            GMX_THROW(InconsistentInputError(
                    "Adaptive NEP/MM requires 0 <= inner cutoff <= outer cutoff."));
        }
    }

    // Create temporary index for the whole System
    auto systemIndices = indexGroupsAndNames.indices("System");

    // Sort system indices
    std::sort(systemIndices.begin(), systemIndices.end());

    // Create MM index
    params_.mmIndices_.reserve(systemIndices.size());

    // Position in nnpIndices_
    size_t j = 0;
    // Write to mmIndices_ only the atoms which do not belong to NNP input region
    for (size_t i = 0; i < systemIndices.size(); i++)
    {
        if (systemIndices[i] != params_.nnpIndices_[j])
        {
            params_.mmIndices_.push_back(systemIndices[i]);
        }
        else
        {
            if (j < params_.nnpIndices_.size() - 1)
            {
                j++;
            }
        }
    }
}

void NNPotOptions::setLocalInputAtomSet(const LocalAtomSet& localInputAtomSet)
{
    params_.nnpAtoms_ = std::make_unique<LocalAtomSet>(localInputAtomSet);
}

void NNPotOptions::setLocalMMAtomSet(const LocalAtomSet& localMMAtomSet)
{
    params_.mmAtoms_ = std::make_unique<LocalAtomSet>(localMMAtomSet);
}

void NNPotOptions::modifyTopology(gmx_mtop_t* top)
{
    // Exit if module is not active
    if (!params_.active_)
    {
        return;
    }

    if (params_.gromacsCharges_)
    {
        if (params_.adaptiveMode_ != NNPotAdaptiveMode::Off)
        {
            GMX_THROW(InconsistentInputError(
                    "nnpot-gromacs-charges cannot be combined with adaptive NEP/MM."));
        }
        if (static_cast<int>(params_.nnpIndices_.size()) != top->natoms)
        {
            GMX_THROW(NotImplementedError(
                    "nnpot-gromacs-charges currently supports a full-system qNEP only. "
                    "The input group must contain all atoms."));
        }
        if (top->ffparams.fudgeQQ != 1.0)
        {
            GMX_THROW(InconsistentInputError(
                    "nnpot-gromacs-charges requires fudgeQQ = 1 so that every atom "
                    "pair interacts with the full dynamic-charge Coulomb term."));
        }
        // Disable classical LJ (the qNEP model provides the short-range
        // interactions) while keeping the topology charges as placeholders;
        // they are overwritten with the dynamic qNEP charges every step.
        params_.linkFrontier_ = preprocessNNPotTopology(top,
                                                        params_.nnpIndices_,
                                                        params_.embeddingScheme_,
                                                        params_.nnpCharge_,
                                                        true,
                                                        *logger_,
                                                        wi_);
        // Every pair must interact through GROMACS Coulomb/PME.
        clearExclusionsForQnepGromacs(top);
        GMX_LOG(logger().info)
                .appendText("qNEP with GROMACS-owned electrostatics: all exclusions and "
                            "scaled 1-4 interactions were removed; dynamic charges are "
                            "predicted every step.");
        top->finalize();
        return;
    }

    if (params_.adaptiveMode_ != NNPotAdaptiveMode::Off)
    {
        std::vector<int> moleculeStarts;
        int              moleculeBlock = 0;
        for (const Index atomIndex : params_.adaptiveBIndices_)
        {
            int moleculeIndex = 0;
            mtopGetMolblockIndex(
                    *top, atomIndex, &moleculeBlock, &moleculeIndex, nullptr);
            const auto& block = top->moleculeBlockIndices[moleculeBlock];
            moleculeStarts.push_back(block.globalAtomStart
                                     + moleculeIndex * block.numAtomsPerMolecule);
        }
        std::sort(moleculeStarts.begin(), moleculeStarts.end());
        moleculeStarts.erase(
                std::unique(moleculeStarts.begin(), moleculeStarts.end()),
                moleculeStarts.end());

        params_.adaptiveBMolecules_.clear();
        params_.adaptiveBIndices_.clear();
        moleculeBlock = 0;
        for (const int moleculeStart : moleculeStarts)
        {
            int moleculeIndex = 0;
            mtopGetMolblockIndex(
                    *top, moleculeStart, &moleculeBlock, &moleculeIndex, nullptr);
            const int moleculeSize =
                    top->moleculeBlockIndices[moleculeBlock].numAtomsPerMolecule;
            std::vector<Index> molecule;
            molecule.reserve(moleculeSize);
            for (int atom = 0; atom < moleculeSize; ++atom)
            {
                if (std::binary_search(params_.nnpIndices_.begin(),
                                       params_.nnpIndices_.end(),
                                       moleculeStart + atom))
                {
                    GMX_THROW(InconsistentInputError(
                            "Expanding adaptive group B to complete molecules overlaps group A."));
                }
                molecule.push_back(moleculeStart + atom);
                params_.adaptiveBIndices_.push_back(moleculeStart + atom);
            }
            params_.adaptiveBMolecules_.push_back(std::move(molecule));
        }

        if (params_.adaptiveMode_ == NNPotAdaptiveMode::CrossOnly)
        {
            GMX_LOG(logger().info)
                    .appendTextFormatted(
                            "Adaptive cross-only NEP/MM: MM topology is unchanged; "
                            "%zu complete B molecules are eligible for delta-NEP corrections.",
                            params_.adaptiveBMolecules_.size());
            return;
        }
    }

    params_.linkFrontier_ = preprocessNNPotTopology(top,
                                                    params_.nnpIndices_,
                                                    params_.embeddingScheme_,
                                                    params_.nnpCharge_,
                                                    params_.useGromacsElectrostatics_,
                                                    *logger_,
                                                    wi_);
}

void NNPotOptions::adjustInputrec(t_inputrec* inputrec)
{
#if GMX_NEP
    if (!params_.active_ || inputrec == nullptr)
    {
        return;
    }

    const std::filesystem::path modelPath(params_.modelFileName_);
    const bool isNepModel =
            (modelPath.extension() == ".txt" || modelPath.extension() == ".nep");
    if (params_.gromacsCharges_)
    {
        // The qNEP model provides the dynamic charges and GROMACS owns the
        // electrostatics, so PME must stay active with the standard settings.
        if (inputrec->coulombtype != CoulombInteractionType::Pme)
        {
            GMX_THROW(InconsistentInputError(
                    "nnpot-gromacs-charges requires coulombtype = PME."));
        }
        if (inputrec->pme_order != 4)
        {
            GMX_THROW(NotImplementedError(
                    "nnpot-gromacs-charges currently supports pme-order = 4 only."));
        }
        if (inputrec->efep != FreeEnergyPerturbationType::No)
        {
            GMX_THROW(NotImplementedError(
                    "nnpot-gromacs-charges does not support free-energy perturbation."));
        }
        if (!inputrec->mtsLevels.empty())
        {
            GMX_THROW(NotImplementedError(
                    "nnpot-gromacs-charges does not support multiple time stepping."));
        }
        if (std::fabs(inputrec->rcoulomb - inputrec->rlist) > 1e-5
            || inputrec->rvdw > inputrec->rcoulomb)
        {
            GMX_THROW(InconsistentInputError(
                    "nnpot-gromacs-charges requires rcoulomb = rlist and "
                    "rvdw <= rcoulomb so that the short-range potential neighbor "
                    "list matches the GROMACS Coulomb pair list."));
        }
        GMX_LOG(logger().info)
                .appendText("qNEP with GROMACS-owned electrostatics: PME stays active and "
                            "receives the dynamic charges every step.");
        return;
    }
    // At this point modifyTopology() has already seen the complete topology.
    // Only a genuine full-system NEP has no classical electrostatics left, so
    // scheduling PME would launch zero-contribution GPU work every step.
    if (isNepModel && params_.adaptiveMode_ == NNPotAdaptiveMode::Off
        && params_.mmIndices_.empty() && !params_.useGromacsElectrostatics_)
    {
        inputrec->coulombtype = CoulombInteractionType::Cut;
        GMX_LOG(logger().info)
                .appendText("Full-system NEP detected: disabling zero-contribution PME "
                            "scheduling. NEP-MM systems retain their configured electrostatics.");
    }
#else
    GMX_UNUSED_VALUE(inputrec);
#endif
}

void NNPotOptions::writeParamsToKvt(KeyValueTreeObjectBuilder treeBuilder)
{
    // Check if active
    if (!params_.active_)
    {
        return;
    }

    // Write input atom indices
    auto GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(moduleName() + "-" + c_inputGroupTag_);
    for (const auto& indexValue : params_.nnpIndices_)
    {
        GroupIndexAdder.addValue(indexValue);
    }

    // Write MM atoms index
    GroupIndexAdder = treeBuilder.addUniformArray<std::int64_t>(moduleName() + "-" + c_mmGroupTag_);
    for (const auto& indexValue : params_.mmIndices_)
    {
        GroupIndexAdder.addValue(indexValue);
    }

    // Write link
    GroupIndexAdder = treeBuilder.addUniformArray<std::int64_t>(moduleName() + "-" + c_nnpLinkTag_);
    for (const auto& link : params_.linkFrontier_)
    {
        GroupIndexAdder.addValue(link.getEmbeddedIndex());
    }
    GroupIndexAdder = treeBuilder.addUniformArray<std::int64_t>(moduleName() + "-" + c_mmLinkTag_);
    for (const auto& link : params_.linkFrontier_)
    {
        GroupIndexAdder.addValue(link.getMMIndex());
    }

    GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(moduleName() + "-" + c_adaptiveBGroupTag_);
    for (const auto indexValue : params_.adaptiveBIndices_)
    {
        GroupIndexAdder.addValue(indexValue);
    }
    auto moleculeValueAdder = treeBuilder.addUniformArray<std::int64_t>(
            moduleName() + "-" + c_adaptiveBMoleculeStartsTag_);
    auto moleculeSizeAdder = treeBuilder.addUniformArray<std::int64_t>(
            moduleName() + "-" + c_adaptiveBMoleculeSizesTag_);
    for (const auto& molecule : params_.adaptiveBMolecules_)
    {
        moleculeValueAdder.addValue(molecule.front());
        moleculeSizeAdder.addValue(molecule.size());
    }

    // check that the model and model inputs are valid
    checkNNPotModel();
}

void NNPotOptions::readParamsFromKvt(const KeyValueTreeObject& tree)
{
    // Check if active
    if (!params_.active_)
    {
        return;
    }

    // Try to read input atoms index
    std::string key = moduleName() + "-" + c_inputGroupTag_;
    if (!tree.keyExists(key))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find input atoms index vector required for neural network potential.\n"
                "This could be caused by incompatible or corrupted tpr input file."));
    }
    auto kvtIndexArray = tree[key].asArray().values();
    params_.nnpIndices_.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray),
                   std::end(kvtIndexArray),
                   std::begin(params_.nnpIndices_),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    // Try to read MM atoms index
    key = moduleName() + "-" + c_mmGroupTag_;
    if (!tree.keyExists(key))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find MM atoms index vector required for neural network potential.\n"
                "This could be caused by incompatible or corrupted tpr input file."));
    }
    kvtIndexArray = tree[key].asArray().values();
    params_.mmIndices_.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray),
                   std::end(kvtIndexArray),
                   std::begin(params_.mmIndices_),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    key = moduleName() + "-" + c_adaptiveBGroupTag_;
    if (tree.keyExists(key))
    {
        kvtIndexArray = tree[key].asArray().values();
        params_.adaptiveBIndices_.resize(kvtIndexArray.size());
        std::transform(std::begin(kvtIndexArray),
                       std::end(kvtIndexArray),
                       std::begin(params_.adaptiveBIndices_),
                       [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });
    }
    const std::string moleculeStartsKey =
            moduleName() + "-" + c_adaptiveBMoleculeStartsTag_;
    const std::string moleculeSizesKey =
            moduleName() + "-" + c_adaptiveBMoleculeSizesTag_;
    if (tree.keyExists(moleculeStartsKey) && tree.keyExists(moleculeSizesKey))
    {
        const auto starts = tree[moleculeStartsKey].asArray().values();
        const auto sizes  = tree[moleculeSizesKey].asArray().values();
        GMX_RELEASE_ASSERT(starts.size() == sizes.size(),
                           "Adaptive molecule start/size arrays differ.");
        params_.adaptiveBMolecules_.clear();
        for (size_t molecule = 0; molecule < starts.size(); ++molecule)
        {
            const int start = starts[molecule].cast<std::int64_t>();
            const int size  = sizes[molecule].cast<std::int64_t>();
            std::vector<Index> indices(size);
            std::iota(indices.begin(), indices.end(), start);
            params_.adaptiveBMolecules_.push_back(std::move(indices));
        }
    }

    // Try to read Link Frontier (two separate vectors and then combine)
    std::vector<Index> nnpLink;
    std::vector<Index> mmLink;

    if (!tree.keyExists(moduleName() + "-" + c_nnpLinkTag_))
    {
        GMX_THROW(
                InconsistentInputError("Cannot find NNP Link Frontier vector required for QM/MM "
                                       "simulation.\nThis could be "
                                       "caused by incompatible or corrupted tpr input file."));
    }
    kvtIndexArray = tree[moduleName() + "-" + c_nnpLinkTag_].asArray().values();
    nnpLink.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray),
                   std::end(kvtIndexArray),
                   std::begin(nnpLink),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    if (!tree.keyExists(moduleName() + "-" + c_mmLinkTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find MM Link Frontier vector required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    kvtIndexArray = tree[moduleName() + "-" + c_mmLinkTag_].asArray().values();
    mmLink.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray),
                   std::end(kvtIndexArray),
                   std::begin(mmLink),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    params_.linkFrontier_.reserve(nnpLink.size());
    for (size_t i = 0; i < nnpLink.size(); i++)
    {
        params_.linkFrontier_.emplace_back(nnpLink[i], mmLink[i]);
        params_.linkFrontier_.back().setLinkDistance(params_.linkDistance_);
        AtomProperties atomProp;
        const int      linkAtomNumber = atomProp.atomNumberFromElement(params_.linkType_.c_str());
        GMX_RELEASE_ASSERT(
                linkAtomNumber != -1,
                formatString("Unrecognized link atom type symbol: %s", params_.linkType_.c_str()).c_str());
        params_.linkFrontier_.back().setLinkAtomNumber(linkAtomNumber);
    }

    // add MM link atoms to nnp index
    for (const auto& link : params_.linkFrontier_)
    {
        params_.nnpIndices_.push_back(link.getMMIndex());
    }
}

const NNPotParameters& NNPotOptions::parameters()
{
    return params_;
}

void NNPotOptions::setTopology(const gmx_mtop_t& top)
{
    params_.atoms_    = gmx_mtop_global_atoms(top);
    params_.numAtoms_ = params_.atoms_.nr;
}

void NNPotOptions::setPbcType(const PbcType& pbcType)
{
    params_.pbcType_ = std::make_unique<PbcType>(pbcType);
}

void NNPotOptions::setLogger(const MDLogger& logger)
{
    logger_ = &logger;
}

void NNPotOptions::setComm(const MpiComm& mpiComm)
{
    mpiComm_ = &mpiComm;
}

const MDLogger& NNPotOptions::logger() const
{
    GMX_RELEASE_ASSERT(logger_, "Logger not set for NNPotOptions.");
    return *logger_;
}

const MpiComm& NNPotOptions::mpiComm() const
{
    GMX_RELEASE_ASSERT(mpiComm_, "MPI communicator not set for NNPotOptions.");
    return *mpiComm_;
}

void NNPotOptions::setWarninp(WarningHandler* wi)
{
    wi_ = wi;
}

#if !GMX_TORCH && !GMX_NEP
[[noreturn]]
#endif
void NNPotOptions::checkNNPotModel()
{
#if GMX_NEP
    const std::filesystem::path modelPath(params_.modelFileName_);
    if (modelPath.extension() == ".txt" || modelPath.extension() == ".nep")
    {
        const std::string header = readModelHeader(params_.modelFileName_);
        if (params_.adaptiveMode_ != NNPotAdaptiveMode::Off)
        {
            const std::string adaptiveHeader =
                    readModelHeader(params_.adaptiveModelFileName_);
            if (isQnepHeader(adaptiveHeader))
            {
                GMX_THROW(InconsistentInputError(
                        "Adaptive NEP/MM does not support qNEP models. Dynamic qNEP "
                        "charge neutralization and long-range electrostatics depend on "
                        "the selected atom set. Use a plain delta-NEP model."));
            }
        }
        if (isQnepHeader(header) && params_.useGromacsElectrostatics_)
        {
            GMX_THROW(InconsistentInputError(
                    "A native qNEP charge1/charge2 model already contains dynamic-charge "
                    "electrostatics and its charge-response chain-rule forces. Enabling "
                    "nnpot-use-gromacs-electrostatics would double count electrostatics. "
                    "Use native qNEP mode, or train a plain short-range residual NEP against "
                    "energies/forces with fixed-charge electrostatics removed."));
        }
        if (params_.gromacsCharges_)
        {
            if (!isQnepHeader(header))
            {
                GMX_THROW(InconsistentInputError(
                        "nnpot-gromacs-charges requires a qNEP charge1/charge2 model; "
                        "a plain NEP has no dynamic charges."));
            }
            if (params_.useGromacsElectrostatics_)
            {
                GMX_THROW(InconsistentInputError(
                        "nnpot-gromacs-charges and nnpot-use-gromacs-electrostatics are "
                        "mutually exclusive."));
            }
        }
        if (params_.embeddingScheme_ != NNPotEmbedding::Mechanical)
        {
            GMX_THROW(NotImplementedError(
                    "NEP/MM currently supports mechanical embedding only."));
        }
        MpiComm comm(MpiComm::SingleRank{});
        NepModel model(params_.modelFileName_, logger(), comm);
        GMX_LOG(logger().info).appendText(
                "Using GPUMD NEP backend (CUDA) for the NNP region.");
        return;
    }
#endif
#if GMX_TORCH

    // try initializing the neural network model
    std::filesystem::path        modelPath(params_.modelFileName_);
    std::unique_ptr<INNPotModel> model;
    MpiComm                      comm(MpiComm::SingleRank{});
    if (!std::filesystem::exists(modelPath))
    {
        GMX_THROW(FileIOError("Model file does not exist: " + params_.modelFileName_));
    }
    else if (modelPath.extension() == ".pt")
    {
        model = std::make_unique<TorchModel>(
                params_.modelFileName_, params_.embeddingScheme_, logger(), comm);
    }
    else
    {
        GMX_THROW(FileIOError("Unrecognized extension for model file: " + params_.modelFileName_));
    }

    // check that link atom type is valid
    AtomProperties atomProp;
    const int      linkAtomNumber = atomProp.atomNumberFromElement(params_.linkType_.c_str());
    if (linkAtomNumber == -1)
    {
        GMX_THROW(InconsistentInputError(
                formatString("Unrecognized link atom type symbol: %s", params_.linkType_.c_str())));
    }

    // check if model accepts inputs
    // Prepare dummy input with two atoms in the NN model, so we can check the pairlist inputs
    std::vector<int>  indices(1, 0);
    std::vector<RVec> positions(2, RVec({ 0.0, 0.0, 0.0 }));
    std::vector<int> atomNumbers(2, linkAtomNumber); // to check that the model can accept the link atom type
    std::vector<int>              atomPairs{ 0, 1 };
    std::vector<RVec>             pairShifts{ RVec({ 0.0, 0.0, 0.0 }) };
    std::vector<real>             charges(1, 1.0);
    matrix                        box = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    PbcType                       pbc = PbcType();
    std::vector<int>              mmIndices(1, 1);
    std::vector<LinkFrontierAtom> link;

    // check that inputs are not empty
    if (params_.modelInput_.size() == 0)
    {
        GMX_THROW(InconsistentInputError("No inputs to NN model provided."));
    }
    // check that sensible cutoff was provided
    const bool pairlistNeeded =
            params_.modelNeedsInput("atom-pairs") || params_.modelNeedsInput("pair-shifts");
    if (pairlistNeeded && (params_.pairCutoff_ <= 0.0))
    {
        GMX_THROW(InconsistentInputError(formatString(
                "List of atom pairs was requested as input to the NNP model, but pair-cutoff is "
                "%.1f. Please specify a valid cutoff radius or disable pair input.",
                params_.pairCutoff_)));
    }

    const auto& theLogger = logger();
    // Check that warning handler is valid
    GMX_ASSERT(wi_, "WarningHandler not set.");
    bool modelOutputsForces = false;
    try
    {
        // prepare dummy output (1 NNP and 1 MM atom)
        gmx_enerdata_t    enerd(2, nullptr);
        std::vector<RVec> forcesVec(2, RVec({ 0.0, 0.0, 0.0 }));
        ArrayRef<RVec>    forces(forcesVec);

        // might throw a runtime error if the model is not compatible with the dummy input
        model->evaluateModel(&enerd,
                             forces,
                             indices,
                             mmIndices,
                             params_.modelInput_,
                             positions,
                             atomNumbers,
                             atomPairs,
                             pairShifts,
                             positions,
                             charges,
                             params_.nnpCharge_,
                             link,
                             &box,
                             &pbc,
                             nullptr);

        // check if model outputs forces after forward pass
        modelOutputsForces = model->outputsForces();

        // log wheter model outputs forces or we need to compute them
        if (modelOutputsForces)
        {
            GMX_LOG(theLogger.info).appendText("Will use forces from NNP model.");
        }
        else
        {
            if (params_.embeddingScheme_ == NNPotEmbedding::ElectrostaticModel)
            {
                GMX_THROW(
                        InconsistentInputError("NNP model does not output forces, but they are "
                                               "expected to be computed by the model for "
                                               "electrostatic embedding scheme."));
            }
            GMX_LOG(theLogger.info)
                    .appendText(
                            "NNP model does not output forces. They will be computed as gradients "
                            "of the energy w.r.t. the first input tensor (atom positions).");
        }
    }
    catch (const std::exception& e)
    {
        // we only issue a warning here instead of throwing an error, as a torch runtime error might
        // simply be due to mismatched dummy input shapes
        wi_->addWarning("There was an error while checking NN model with a dummy input: " + std::string(e.what()) + "\n"
                        "Can't verify that the model works correctly. This might lead to errors during mdrun.");
    }

    // check if first input is atom-positions
    if ((params_.modelInput_[0] != "atom-positions") && !modelOutputsForces)
    {
        wi_->addWarning("Gradients will be computed with respect to first input to NN model "
                        + params_.modelInput_[0] + " instead of atom positions. Is this intended?");
    }

    // check if model might need MM atom positions and/or charges
    if (params_.embeddingScheme_ == NNPotEmbedding::ElectrostaticModel)
    {
        if (!params_.modelNeedsInput("atom-positions-mm") && !params_.modelNeedsInput("atom-charges-mm"))
        {
            wi_->addWarning(
                    "Embedding scheme is set to Electrostatic model, but MM positions and/or "
                    "charges are not requested as model input. Is this intended?");
        }
    }

#else
    GMX_THROW(InternalError(
            "Libtorch/NN backend is not linked into GROMACS, NNPot simulation is not possible."
            " Please, reconfigure GROMACS with -DGMX_NNPOT=TORCH\n"));
#endif // GMX_TORCH
}

} // namespace gmx
