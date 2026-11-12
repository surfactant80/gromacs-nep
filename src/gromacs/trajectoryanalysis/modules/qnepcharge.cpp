/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */
/*! \internal \file
 * \brief
 * Implements the gmx qnep-charge trajectory analysis module.
 *
 * The module reads a trajectory together with a topology and a GPUMD qNEP
 * model file, predicts the dynamic atomic charges for every frame with the
 * vendored GPUMD kernels on the GPU (no external process, no modification of
 * the GPUMD source), and writes the selected atoms' charges as a function of
 * time. Optional per-frame dipole moments can be enabled with -dipole.
 */
#include "qnepcharge.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "gromacs/applied_forces/nnpot/nepmodel.h"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selection.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/selection/selectioncollection.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/mtop_atomloops.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/fatalerror.h"

namespace gmx::analysismodules
{

//! Frame-local data holder for the charge module.
class QnepChargeModuleData : public TrajectoryAnalysisModuleData
{
public:
    QnepChargeModuleData(TrajectoryAnalysisModule*          module,
                     const AnalysisDataParallelOptions& opt,
                     const SelectionCollection&         selections) :
        TrajectoryAnalysisModuleData(module, opt, selections)
    {
    }
    void finish() override { finishDataHandles(); }
};

class QnepCharge : public TrajectoryAnalysisModule
{
public:
    QnepCharge();

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void initAfterFirstFrame(const TrajectoryAnalysisSettings& settings, const t_trxframe& fr) override;
    TrajectoryAnalysisModuleDataPointer startFrames(const AnalysisDataParallelOptions& opt,
                                                     const SelectionCollection& selections) override;
    void analyzeFrame(int                           frnr,
                      const t_trxframe&            fr,
                      t_pbc*                       pbc,
                      TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    Selection selection_;
    std::string nepFile_ = "";
    real        rcoulomb_ = 0.8;
    bool        doDipole_ = false;
    real        epsInf_   = 1.78;
    real        chargeScale_ = 1.0;
    std::string output_   = "qnep-charge";
    std::string outputDipole_ = "dipole.xvg";
    AnalysisDataPlotSettings plotSettings_;

    //! Dipole moment plot data (only used with -dipole).
    AnalysisData dipoleData_;

    //! GPU context kept alive for the whole run.
    std::unique_ptr<DeviceContext> gpuContext_;
    //! qNEP model wrapper (uses the vendored GPUMD kernels).
    std::shared_ptr<gmx::NepModel> nepModel_;
    //! Atomic numbers in the topology order (used to map the model types).
    std::vector<int> atomNumbers_;
    //! Selected atom indices.
    std::vector<int> selIndices_;
    //! Atomic numbers covered by the qNEP model (the output region).
    std::vector<int> coveredZ_;

    //! Plot data.
    AnalysisData plotData_;
};

QnepCharge::QnepCharge() = default;

void QnepCharge::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] predicts and outputs the dynamic atomic charges of a GPUMD qNEP",
        "model for every frame of a trajectory.\n",
        "A GPUMD qNEP model is required ([TT]-nep[tt]); the charges are computed on the",
        "GPU with the vendored GPUMD kernels (the same prediction as during mdrun).",
        "The qNEP charges are latent charges: to obtain the physical point charges they",
        "are multiplied by sqrt(eps_inf), with eps_inf the high-frequency relative",
        "permittivity of the material ([TT]-epsinf[tt], default 1.78 for water).",
        "By default all atoms of the qNEP region are output; use [TT]-select[tt] to",
        "restrict the output to a group (e.g. the qNEP region).",
        "With [TT]-dipole[tt], a second output file ([TT]-odip[tt]) is written with the",
        "total dipole moment of the selected atoms as a function of time (the charge",
        "output file then contains only charges).",
    };
    settings->setHelpText(desc);

    // The default selection is the region covered by the qNEP model; the
    // actual selection text is filled in initAnalysis() once the model's
    // element set is known (the option default is "none" until then).
    options->addOption(SelectionOption("sel")
                               .store(&selection_)
                               .defaultSelectionText("all")
                               .onlyAtoms()
                               .description("Atoms to output charges for "
                                            "(default: the region covered by the qNEP model)"));
    options->addOption(StringOption("nep")
                               .store(&nepFile_)
                               .required()
                               .description("GPUMD qNEP model file (nep4_charge1/charge2)"));
    options->addOption(RealOption("rcoulomb")
                               .store(&rcoulomb_)
                               .defaultValue(0.8)
                               .description("Coulomb cutoff (nm) used for the charge prediction"));
    options->addOption(RealOption("epsinf")
                               .store(&epsInf_)
                               .defaultValue(1.78)
                               .description("High-frequency relative permittivity of the material "
                                            "(the latent charges are scaled by sqrt(eps_inf)); "
                                            "default 1.78 for water"));
    options->addOption(BooleanOption("dipole")
                               .store(&doDipole_)
                               .defaultValue(false)
                               .description("Also output the total dipole moment in Debye to -odip"));
    options->addOption(FileNameOption("odip")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&outputDipole_)
                               .defaultBasename("dipole")
                               .description("Dipole moment output file (used with -dipole)"));
    options->addOption(FileNameOption("o")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&output_)
                               .defaultBasename("qnep-charge")
                               .description("Output file name"));
}

void QnepCharge::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    plotSettings_ = settings.plotSettings();
    if (!gpuContext_)
    {
        // Activate the first CUDA device for the vendored GPUMD kernels. Only
        // the device id is required by the CUDA implementation.
        DeviceInformation devInfo{};
        devInfo.id           = 0;
        devInfo.deviceVendor = DeviceVendor::Nvidia;
        gpuContext_          = std::make_unique<DeviceContext>(devInfo);
        gpuContext_->activate();
        fprintf(stderr, "charge: using GPU #0 for the qNEP charge prediction\n");
    }
    MDLogger logger;
    MpiComm  comm(MpiComm::SingleRank{});
    // The analysis tool evaluates charges through the GPUMD descriptor
    // kernels, which WPE does not expose, so this is always the GPUMD backend.
    gmx::NepBackendSettings nepSettings;
    nepSettings.backend = gmx::NepBackend::Gpumd;
    nepModel_ = std::make_shared<gmx::NepModel>(nepFile_, logger, comm, false, nepSettings);
    coveredZ_ = nepModel_->coveredAtomicNumbers();
    chargeScale_ = static_cast<real>(std::sqrt(static_cast<double>(epsInf_)));

    atomNumbers_.clear();
    if (top.atoms() != nullptr)
    {
        const t_atoms&  atoms = *top.atoms();
        atomNumbers_.resize(atoms.nr);
        for (int i = 0; i < atoms.nr; ++i)
        {
            atomNumbers_[i] = atoms.atom[i].atomnumber;
        }
    }
}

void QnepCharge::initAfterFirstFrame(const TrajectoryAnalysisSettings& settings, const t_trxframe& fr)
{
    (void)settings;
    if (static_cast<int>(atomNumbers_.size()) != fr.natoms)
    {
        gmx_fatal(FARGS, "The topology contains %zu atoms but the trajectory frame has %d.",
                  atomNumbers_.size(), fr.natoms);
    }
    selIndices_.clear();
    for (const int index : selection_.atomIndices())
    {
        if (index < 0 || index >= fr.natoms)
        {
            continue;
        }
        const int z = atomNumbers_[index];
        if (std::find(coveredZ_.begin(), coveredZ_.end(), z) == coveredZ_.end())
        {
            // Atom is outside the qNEP model region: its charge is the fixed
            // topology value, so it is not part of the qNEP charge output.
            continue;
        }
        selIndices_.push_back(index);
    }
    registerAnalysisDataset(&plotData_, "charges");
    if (doDipole_)
    {
        registerAnalysisDataset(&dipoleData_, "dipole");
    }
}

TrajectoryAnalysisModuleDataPointer QnepCharge::startFrames(
        const AnalysisDataParallelOptions& opt, const SelectionCollection& selections)
{
    const int nsel  = static_cast<int>(selIndices_.size());
    AnalysisDataPlotModulePointer plotModule(new AnalysisDataPlotModule(plotSettings_));
    plotModule->setFileName(output_);
    plotModule->setTitle("qNEP dynamic charges");
    plotModule->setXLabel("time");
    plotModule->setXAxisIsTime();
    plotModule->setYLabel("charge (e)");
    plotModule->setYFormat(9, 4, 'g');
    plotData_.addModule(plotModule);
    plotData_.setDataSetCount(1);
    plotData_.setColumnCount(0, nsel + 2);
    if (doDipole_)
    {
        AnalysisDataPlotModulePointer dipModule(new AnalysisDataPlotModule(plotSettings_));
        dipModule->setFileName(outputDipole_);
        dipModule->setTitle("Dipole moment");
        dipModule->setXLabel("time");
        dipModule->setXAxisIsTime();
        dipModule->setYLabel("dipole (Debye)");
        dipModule->setYFormat(9, 4, 'g');
        dipoleData_.addModule(dipModule);
        dipoleData_.setDataSetCount(1);
        dipoleData_.setColumnCount(0, 4);
    }
    return TrajectoryAnalysisModuleDataPointer(new QnepChargeModuleData(this, opt, selections));
}

void QnepCharge::analyzeFrame(int                           frnr,
                          const t_trxframe&            fr,
                          t_pbc*                       pbc,
                          TrajectoryAnalysisModuleData* pdata)
{
    (void)pbc;
    if (fr.x == nullptr || fr.box == nullptr || fr.natoms <= 0)
    {
        return;
    }
    std::vector<RVec> frameX(fr.natoms);
    for (int i = 0; i < fr.natoms; ++i)
    {
        frameX[i] = { fr.x[i][XX], fr.x[i][YY], fr.x[i][ZZ] };
    }
    std::vector<float> charges;
    if (!nepModel_ || !nepModel_->predictChargesHostFrame(
                              frameX, fr.box, fr.natoms, rcoulomb_, atomNumbers_, charges))
    {
        gmx_fatal(FARGS, "qNEP charge prediction failed (is the model a qNEP charge model?)");
    }

    const double time = fr.time;
    AnalysisDataHandle dh = pdata->dataHandle(plotData_);
    dh.startFrame(frnr, time);
    dh.selectDataSet(0);
    const double scale = static_cast<double>(chargeScale_);
    double mean = 0.0, sum = 0.0;
    size_t col = 0;
    for (const int idx : selIndices_)
    {
        const double q = (idx >= 0 && idx < fr.natoms) ? charges[idx] * scale : 0.0;
        dh.setPoint(col++, q);
        mean += q;
        sum += q;
    }
    if (!selIndices_.empty())
    {
        mean /= selIndices_.size();
    }
    dh.setPoint(col++, mean);
    dh.setPoint(col++, sum);
    dh.finishFrame();
    if (doDipole_)
    {
        double mx = 0.0, my = 0.0, mz = 0.0;
        for (const int idx : selIndices_)
        {
            if (idx < 0 || idx >= fr.natoms)
            {
                continue;
            }
            mx += charges[idx] * scale * fr.x[idx][XX];
            my += charges[idx] * scale * fr.x[idx][YY];
            mz += charges[idx] * scale * fr.x[idx][ZZ];
        }
        // The wrapped-coordinate dipole moment of a periodic system (the
        // standard "wrapped" definition together with the sum rule).
        // Convert e*nm to Debye: 1 e*nm = 48.03 D.
        constexpr double c_eNmToDebye = 48.0313;
        mx *= c_eNmToDebye;
        my *= c_eNmToDebye;
        mz *= c_eNmToDebye;
        const double m2 = std::sqrt(mx * mx + my * my + mz * mz);
        AnalysisDataHandle ddh = pdata->dataHandle(dipoleData_);
        ddh.startFrame(frnr, time);
        ddh.selectDataSet(0);
        ddh.setPoint(0, mx);
        ddh.setPoint(1, my);
        ddh.setPoint(2, mz);
        ddh.setPoint(3, m2);
        ddh.finishFrame();
    }
}

void QnepCharge::finishAnalysis(int nframes)
{
    (void)nframes;
}

void QnepCharge::writeOutput()
{
}

const char                      QnepChargeInfo::name[] = "qnep-charge";
const char                      QnepChargeInfo::shortDescription[] = "Output qNEP dynamic atomic charges";
TrajectoryAnalysisModulePointer QnepChargeInfo::create()
{
    return TrajectoryAnalysisModulePointer(new QnepCharge);
}

} // namespace gmx::analysismodules
