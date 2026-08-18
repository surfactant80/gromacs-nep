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
 * Declares the options for NNPot MDModule class,
 * set during pre-processing in the .mdp-file.
 *
 * \author Lukas Müllender <lukas.muellender@gmail.com>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_NNPOTOPTIONS_H
#define GMX_APPLIED_FORCES_NNPOTOPTIONS_H

#include "config.h"

#include <string>
#include <vector>

#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/vectypes.h"

// some forward declarations
struct gmx_mtop_t;
struct t_inputrec;
class WarningHandler;
enum class PbcType;

namespace gmx
{

// more forward declarations
class MpiComm;
class MDLogger;
class IOptionsContainerWithSections;
class IKeyValueTreeTransformRules;
class KeyValueTreeObjectBuilder;
class KeyValueTreeObject;
class IndexGroupsAndNames;
class LocalAtomSet;

//! \brief Enum to specify embedding scheme used for NNP/MM interaction
enum class NNPotEmbedding
{
    //! Mechanical embedding: NNP/MM interaction calculated classically
    Mechanical,
    //! Electrostatic embedding calculated within NNP model (e.g.g EMLE-engine)
    ElectrostaticModel,
    // TODO: add more embedding schemes (Polarizable, ...)
    Count
};

//! Runtime adaptive NEP/MM coupling mode.
enum class NNPotAdaptiveMode
{
    //! Static NNPot behavior.
    Off,
    //! Keep A-A and B-B in MM; add a delta-NEP correction for nearby A-B molecules.
    CrossOnly,
    //! Treat A internally with NEP and add a delta-NEP correction for nearby A-B molecules.
    ANepCross,
    Count
};

//!\brief \internal Data structure to store NNPot input parameters
struct NNPotParameters
{
    //! indicates whether NN Potential is active (default false)
    bool active_ = false;

    //! stores file name of NNPot model
    std::string modelFileName_ = "model.pt";

    //! stores atom group name for neural network input (default whole System)
    std::string inputGroup_ = "System";
    //! Indices of the atoms that are part of the NN input region (default whole System)
    std::vector<Index> nnpIndices_;
    //! Local set of atoms that are part of the NN input region
    std::unique_ptr<LocalAtomSet> nnpAtoms_;

    //! Indices of the atoms that are part of the MM region (default no MM atoms)
    std::vector<Index> mmIndices_;
    //! Local set of atoms that are part of the MM region
    std::unique_ptr<LocalAtomSet> mmAtoms_;

    //! User defined input to NN model (9 options as of now)
    std::vector<std::string> modelInput_{ "", "", "", "", "", "", "", "", "" };

    //! User defined cutoff for pairlist (if used)
    real pairCutoff_ = 0.0;

    //! stores pbc type used by the simulation
    std::unique_ptr<PbcType> pbcType_;

    //! stores embedding scheme used for NNP/MM interaction
    NNPotEmbedding embeddingScheme_ = NNPotEmbedding::Mechanical;
    //! stores total charge of NNP region
    real nnpCharge_ = 0.0;
    //! Keep classical GROMACS electrostatics for a full-system residual NEP.
    bool useGromacsElectrostatics_ = false;
    //! qNEP: predict dynamic charges and let GROMACS own the electrostatics.
    bool gromacsCharges_ = false;
    //! Optional per-element constant offsets, e.g. "H:-1.2 C:3.4" in kJ/mol/atom.
    std::string energyOffsets_;

    //! Adaptive A/B coupling mode.
    NNPotAdaptiveMode adaptiveMode_ = NNPotAdaptiveMode::Off;
    //! Candidate B group name for adaptive coupling.
    std::string adaptiveGroupB_;
    //! Delta-NEP model used for adaptive A-B corrections.
    std::string adaptiveModelFileName_;
    //! Adaptive model must be trained on NEP-minus-MM residual labels.
    bool adaptiveModelIsDelta_ = false;
    //! Full-strength and zero-strength switching distances in nm.
    real adaptiveInnerCutoff_ = 0.4;
    real adaptiveOuterCutoff_ = 0.6;
    //! Global B indices and complete B molecules, built from the topology.
    std::vector<Index> adaptiveBIndices_;
    std::vector<std::vector<Index>> adaptiveBMolecules_;

    //! stores all (global) atom info
    t_atoms atoms_;
    int     numAtoms_;

    //! Stores the link frontier information
    std::vector<LinkFrontierAtom> linkFrontier_;
    //! Type of link atom to use (default "H", hydrogen)
    std::string linkType_ = "H";
    //! Distance between link atom and embedded atom (default 0.1 nm)
    real linkDistance_ = 0.1;

    bool modelNeedsInput(const std::string& input) const
    {
        return std::find(modelInput_.begin(), modelInput_.end(), input) != modelInput_.end();
    }
};

class NNPotOptions final : public IMdpOptionProvider
{
public:
    //! Implementation of IMdpOptionProvider method
    void initMdpTransform(IKeyValueTreeTransformRules* /*rules*/) override;

    //! \brief Connects option names and data.
    void initMdpOptions(IOptionsContainerWithSections* /*options*/) override;

    /*! \brief Build mdp parameters for NNPot to be output after pre-processing.
     * \param[in, out] builder the builder for the mdp options output KVT.
     */
    void buildMdpOutput(KeyValueTreeObjectBuilder* /*builder*/) const override;

    //! return active state of NNPot module
    bool isActive() const;

    //! get model file name
    std::string getModelFileName() const;

    //! set atom group for neural network input
    void setInputGroupIndices(const IndexGroupsAndNames& /*indexGroupsAndNames*/);

    //! set local atom set for neural network input during simulation setup
    void setLocalInputAtomSet(const LocalAtomSet& /*localAtomSet*/);

    //! set local MM atom set during simulation setup
    void setLocalMMAtomSet(const LocalAtomSet& /*localMMAtomSet*/);

    //! modify topology of the system during preprocessing
    void modifyTopology(gmx_mtop_t* /*mtop*/);

    //! Adjust run settings after the NNP group and topology are known.
    void adjustInputrec(t_inputrec* /*inputrec*/);

    //! set topology of the system during simulation setup
    void setTopology(const gmx_mtop_t& /*mtop*/);

    //! set communication object during simulation setup
    void setComm(const MpiComm& /*mpiComm*/);

    //! Store the paramers that are not mdp options in the tpr file
    // This is needed to retain data from preprocessing to simulation setup
    void writeParamsToKvt(KeyValueTreeObjectBuilder /*treeBuilder*/);

    //! Set the internal parameters that are stored in the tpr file
    void readParamsFromKvt(const KeyValueTreeObject& /*tree*/);

    //! get NNPot parameters
    const NNPotParameters& parameters();

    //! set PBC type during simulation setup
    void setPbcType(const PbcType& /*pbcType*/);

    void            setLogger(const MDLogger& /*logger*/);
    void            setWarninp(WarningHandler* /*wi*/);
    const MDLogger& logger() const;
    const MpiComm&  mpiComm() const;

private:
//! Make sure that model and model inputs are compatible
#if !GMX_TORCH && !GMX_NEP
    [[noreturn]] static
#endif
            void
            checkNNPotModel();

    //! NNPot parameters
    NNPotParameters params_;

    /*! \brief MDLogger during mdrun
     *
     * This is a pointer only because we need an "optional reference"
     * to a const MDLogger before the notification always provides the
     * actual reference. */
    const MDLogger* logger_ = nullptr;

    //! stores communication object for my group
    const MpiComm* mpiComm_ = nullptr;

    //! Instance of warning bookkeeper
    WarningHandler* wi_ = nullptr;
};

} // namespace gmx

#endif
