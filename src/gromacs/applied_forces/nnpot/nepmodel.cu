#include "gmxpre.h"

#include "nepmodel.h"

#include "gromacs/gpu_utils/device_stream.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/stringutil.h"

#include "nepmodel_qnep_gromacs.h"
#include "wpeloader.h"
#include "wpepotential.h"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/run_input.cuh"
#include "force/nep.cuh"
#include "force/nep_charge.cuh"
#if GMX_NEP_HAVE_VDW
#include "force/nep_charge_vdw/nep_charge_vdw.cuh"
#endif

namespace
{

using gmx::formatString;
using gmx::InconsistentInputError;
using gmx::NepBackend;
using gmx::NepBackendSettings;
using gmx::NepModelFamily;
using gmx::NepModelInfo;
using gmx::WpeStageSettings;

constexpr double c_evToKjMol = 96.48533212331002;
constexpr double c_angstromPerNm = 10.0;
constexpr double c_forceScale = c_evToKjMol * c_angstromPerNm;
constexpr int c_postprocessBlockSize = 256;

template<typename InputReal, bool useIndexLookup>
__global__ void convertPositionsToNep(const int numAtoms,
                                      const InputReal* __restrict__ input,
                                      const int* __restrict__ indexLookup,
                                      double* __restrict__ output)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom < numAtoms)
    {
        const int inputAtom = useIndexLookup ? indexLookup[atom] : atom;
        output[atom] =
                static_cast<double>(input[3 * inputAtom]) * c_angstromPerNm;
        output[numAtoms + atom] =
                static_cast<double>(input[3 * inputAtom + 1]) * c_angstromPerNm;
        output[2 * numAtoms + atom] =
                static_cast<double>(input[3 * inputAtom + 2]) * c_angstromPerNm;
    }
}

template<typename OutputReal, bool useIndexLookup>
__global__ void convertForcesAndReduce(const int numAtoms,
                                       const double* __restrict__ potential,
                                       const double* __restrict__ force,
                                       const double* __restrict__ virial,
                                       const int* __restrict__ indexLookup,
                                       OutputReal* __restrict__ outputForce,
                                       double* __restrict__ summary)
{
    __shared__ double blockSums[10][c_postprocessBlockSize];

    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid  = threadIdx.x;

    double values[10] = {};
    if (atom < numAtoms)
    {
        const int outputAtom = useIndexLookup ? indexLookup[atom] : atom;
        outputForce[3 * outputAtom] =
                static_cast<OutputReal>(force[atom] * c_forceScale);
        outputForce[3 * outputAtom + 1] =
                static_cast<OutputReal>(force[numAtoms + atom] * c_forceScale);
        outputForce[3 * outputAtom + 2] =
                static_cast<OutputReal>(force[2 * numAtoms + atom] * c_forceScale);

        values[0] = potential[atom];
#pragma unroll
        for (int component = 0; component < 9; ++component)
        {
            values[component + 1] = virial[component * numAtoms + atom];
        }
    }

#pragma unroll
    for (int component = 0; component < 10; ++component)
    {
        blockSums[component][tid] = values[component];
    }
    __syncthreads();

    for (int stride = c_postprocessBlockSize / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
#pragma unroll
            for (int component = 0; component < 10; ++component)
            {
                blockSums[component][tid] += blockSums[component][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
#pragma unroll
        for (int component = 0; component < 10; ++component)
        {
            atomicAdd(summary + component, blockSums[component][0]);
        }
    }
}


const char* const c_elements[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"
};

int atomicNumber(const std::string& symbol)
{
    for (int i = 0; i < static_cast<int>(sizeof(c_elements) / sizeof(c_elements[0])); ++i)
    {
        if (symbol == c_elements[i])
        {
            return i + 1;
        }
    }
    return -1;
}

std::string absolutePath(const std::string& filename)
{
    char* resolved = realpath(filename.c_str(), nullptr);
    if (resolved == nullptr)
    {
        throw InconsistentInputError("Could not resolve NEP model path: " + filename);
    }
    std::string result(resolved);
    std::free(resolved);
    return result;
}

/*! \brief Build the GPUMD run.in lines that the NEP constructors consume.
 *
 * GPUMD's potential classes read a few run.in keywords directly (D3 correction
 * and the qNEP reciprocal-space method). GROMACS synthesizes exactly those
 * lines here instead of touching the working directory, so several ranks can
 * create models concurrently and no run.in file is required.
 */
std::vector<std::string> buildGpumdRunInput(const NepModelInfo&       info,
                                            const NepBackendSettings& settings,
                                            const std::string&        modelFile)
{
    std::vector<std::string> lines;
    if (!settings.dftd3.empty())
    {
        if (modelFile.find(' ') != std::string::npos)
        {
            throw InconsistentInputError(
                    "The NEP D3 correction requires a model path without spaces because "
                    "GPUMD's run.in tokenizer splits on whitespace.");
        }
        lines.push_back("potential " + modelFile);
        lines.push_back("dftd3 " + settings.dftd3);
    }
    if (info.family == NepModelFamily::Qnep || info.family == NepModelFamily::Qnep3Vdw)
    {
        if (settings.gpumdKspace == "ewald")
        {
            lines.push_back("kspace ewald");
        }
        else if (settings.gpumdPppmSpacing > 0.0)
        {
            lines.push_back(formatString("kspace pppm %.6g", settings.gpumdPppmSpacing));
        }
        else
        {
            lines.push_back("kspace pppm");
        }
    }
    return lines;
}

/*! \brief Instantiate the vendored GPUMD potential class for a model family. */
std::unique_ptr<Potential> createGpumdModel(const NepModelInfo&       info,
                                            const NepBackendSettings& settings,
                                            const std::string&        modelFile,
                                            const int                 numAtoms)
{
    const RunInput runInput(buildGpumdRunInput(info, settings, modelFile));
    switch (info.family)
    {
        case NepModelFamily::Nep4:
            return std::make_unique<NEP>(modelFile.c_str(), numAtoms, runInput);
        case NepModelFamily::Qnep:
            return std::make_unique<NEP_Charge>(modelFile.c_str(), numAtoms, runInput);
#if GMX_NEP_HAVE_VDW
        case NepModelFamily::NepVdw:
        case NepModelFamily::Qnep3Vdw:
            return std::make_unique<NEP_Charge_VDW>(modelFile.c_str(), numAtoms, runInput);
#endif
        case NepModelFamily::Unsupported:
        default:
            throw InconsistentInputError(
                    "The vendored GPUMD backend cannot evaluate this model: "
                    + info.unsupportedReason);
    }
}

/*! \brief Translate the mdp backend settings into WPE stage facts. */
WpeStageSettings makeWpeStageSettings(const NepModelInfo&       info,
                                      const NepBackendSettings& settings)
{
    WpeStageSettings stage;
    stage.ensembleName      = settings.ensembleName;
    stage.qnepKspaceMethod  = settings.wpeKspace == "ewald" ? WPE_QNEP_KSPACE_EWALD
                                                           : WPE_QNEP_KSPACE_PPPM;
    stage.qnepPppmSpacing   = settings.wpePppmSpacing;
    stage.needChargeOutput  = settings.wantChargeOutput && isQnepModel(info);
    stage.needBecOutput     = false;
    // Debug/compatibility switch: request the expensive 9N atomic virials
    // instead of the default six-component system virial.
    stage.needAtomVirial    = std::getenv("GMX_WPE_ATOM_VIRIAL") != nullptr;

    if (!settings.dftd3.empty())
    {
        std::istringstream stream(settings.dftd3);
        std::string        functional;
        double             rcPotential = 0.0;
        double             rcCoordination = 0.0;
        std::string        extra;
        if (!(stream >> functional >> rcPotential >> rcCoordination) || (stream >> extra))
        {
            throw InconsistentInputError(
                    "nnpot-dftd3 must contain '<functional> <rc_potential> "
                    "<rc_coordination_number>'.");
        }
        stage.dftd3Enabled              = true;
        stage.dftd3Functional           = functional;
        stage.dftd3RcPotential          = static_cast<float>(rcPotential);
        stage.dftd3RcCoordinationNumber = static_cast<float>(rcCoordination);
    }
    return stage;
}

} // namespace

namespace gmx
{

struct NepModel::Impl
{
    explicit Impl(const std::string&        filename,
                  const MDLogger&           logger,
                  const MpiComm&            mpiComm,
                  const bool                gromacsCharges,
                  const NepBackendSettings& settings) :
        modelFile(absolutePath(filename)),
        settings(settings),
        logger(logger),
        mpiComm(mpiComm)
    {
        info             = readNepModelInfo(modelFile);
        isQnep           = isQnepModel(info);
        chargeMode       = info.chargeMode;
        useGromacsCharges = gromacsCharges;

        for (int i = 0; i < info.numTypes; ++i)
        {
            const int z = atomicNumber(info.elements[i]);
            if (z < 0)
            {
                throw InconsistentInputError("Unknown element in NEP model: "
                                             + info.elements[i]);
            }
            typeByAtomicNumber.emplace(z, i);
        }

        resolveBackend();

        if (useGromacsCharges)
        {
            if (backend != NepBackend::Gpumd)
            {
                throw InconsistentInputError(
                        "nnpot-gromacs-charges needs the vendored GPUMD backend because the "
                        "charge-response chain rule reads GPUMD's descriptor kernels. Select "
                        "nnpot-backend = gpumd, or use native qNEP electrostatics with WPE.");
            }
            if (!isQnep)
            {
                throw InconsistentInputError(
                        "nnpot-gromacs-charges requires a qNEP charge1/charge2 model; a plain "
                        "NEP has no dynamic charges.");
            }
            if (info.hasZbl)
            {
                throw InconsistentInputError(
                        "GROMACS-owned qNEP electrostatics does not support ZBL qNEP models.");
            }
            if (info.chargeMode == 3)
            {
                throw InconsistentInputError(
                        "GROMACS-owned qNEP electrostatics supports charge1/charge2 models; "
                        "charge3 models must use native qNEP electrostatics.");
            }
        }
    }

    /*! \brief Decide which engine evaluates the model.
     *
     * This runs during grompp as well, so it must not touch CUDA. WPE library
     * discovery is a filesystem/LD_LIBRARY_PATH search only; the engine is
     * dlopen()ed on the first device evaluation in mdrun.
     */
    void resolveBackend()
    {
        std::string       reason;
        const bool        wpeCompiled = WpeLoader::isCompiledIn();
        const bool        wpeModel    = wpeSupportsModel(info, &reason);
        const std::string wpeLibrary  = WpeLoader::findLibrary(settings.wpeLibrary);
        const bool        wpePresent  = !wpeLibrary.empty();
        std::string       gpumdReason;
        const bool        gpumdModel = gpumdSupportsModel(info, &gpumdReason);

        if (useGromacsCharges)
        {
            backend = NepBackend::Gpumd;
            return;
        }

        switch (settings.backend)
        {
            case NepBackend::Wpe:
                if (!wpeCompiled)
                {
                    throw InconsistentInputError(
                            "nnpot-backend = wpe requires a GROMACS build configured with "
                            "-DGMX_NEP_WPE=ON.");
                }
                if (!wpeModel)
                {
                    throw InconsistentInputError(
                            "The WPE backend cannot evaluate this model: " + reason
                            + " Use nnpot-backend = gpumd for the vendored GPUMD kernels.");
                }
                if (!wpePresent)
                {
                    throw InconsistentInputError(
                            "The WPE backend was requested but libwisevolve_potential.so was "
                            "not found. Set nnpot-wpe-library (or $GMX_WPE_LIBRARY) to the WPE "
                            "runtime directory, or add it to $LD_LIBRARY_PATH.");
                }
                backend = NepBackend::Wpe;
                break;
            case NepBackend::Gpumd:
                if (!gpumdModel)
                {
                    throw InconsistentInputError(
                            "The vendor GPUMD backend cannot evaluate this model: " + gpumdReason);
                }
                backend = NepBackend::Gpumd;
                break;
            case NepBackend::Auto:
            default:
                if (wpeCompiled && wpeModel && wpePresent)
                {
                    backend = NepBackend::Wpe;
                }
                else
                {
                    if (!gpumdModel)
                    {
                        // Neither engine can run this model. Prefer the WPE shape
                        // explanation when WPE was compiled in, since it is the more
                        // specific diagnosis for a WPE-shaped model.
                        throw InconsistentInputError(
                                "No available NEP backend can evaluate this model. GPUMD: "
                                + gpumdReason
                                + (wpeCompiled ? (" WPE: " + reason) : std::string()));
                    }
                    backend = NepBackend::Gpumd;
                    if (wpeCompiled && wpeModel && !wpePresent)
                    {
                        backendNote =
                                "WPE supports this model but libwisevolve_potential.so was not "
                                "found, so the vendored GPUMD backend is used.";
                    }
                    else if (wpeCompiled && !wpeModel)
                    {
                        backendNote = "WPE cannot evaluate this model (" + reason
                                      + "), using the vendored GPUMD backend.";
                    }
                }
                break;
        }
    }

    /*! \brief Create the engine instance on first use (mdrun, after CUDA init). */
    std::unique_ptr<Potential> createModel(const int numAtoms)
    {
        if (backend == NepBackend::Wpe)
        {
            auto wpeModel = std::make_unique<WpePotential>(modelFile,
                                                           numAtoms,
                                                           makeWpeStageSettings(info, settings),
                                                           settings.wpeLibrary);
            GMX_LOG(logger.info)
                    .appendTextFormatted(
                            "Using WPE backend: %s %s (library %s)\n",
                            wpeModel->engineName().c_str(),
                            wpeModel->engineVersion().c_str(),
                            wpeModel->libraryPath().c_str());
            if (!wpeModel->engineCitation().empty())
            {
                GMX_LOG(logger.info)
                        .appendText("WPE citation: " + wpeModel->engineCitation() + "\n");
            }
            return wpeModel;
        }
        return createGpumdModel(info, settings, modelFile, numAtoms);
    }

    ~Impl()
    {
        qnepGromacs_.reset();
        if (completionEvent != nullptr)
        {
            CHECK(cudaEventDestroy(completionEvent));
        }
        if (hostPositions != nullptr)
        {
            CHECK(cudaFreeHost(hostPositions));
        }
        if (hostForces != nullptr)
        {
            CHECK(cudaFreeHost(hostForces));
        }
        if (hostSummary != nullptr)
        {
            CHECK(cudaFreeHost(hostSummary));
        }
        if (hostCharges != nullptr)
        {
            CHECK(cudaFreeHost(hostCharges));
        }
    }

    //! Create the GPUMD/WPE model (and its CUDA buffers) for \p numAtoms.
    void ensureModel(int numAtoms)
    {
        if (model == nullptr || modelAtomCount != numAtoms)
        {
            // Resolve and load the WPE runtime first: this reports a missing
            // proprietary engine or a missing CUDA runtime dependency before
            // the GPU availability check, which is the more actionable error.
            if (backend == NepBackend::Wpe)
            {
                std::string loadError;
                if (WpeLoader::instance(settings.wpeLibrary, &loadError) == nullptr)
                {
                    throw InconsistentInputError(
                            "The WPE backend was selected but the Wisevolve Potential "
                            "Engine could not be loaded: "
                            + loadError);
                }
            }
            // Model construction allocates CUDA buffers, so this is the first
            // point in mdrun where a GPU is required. grompp never gets here.
            int         deviceCount = 0;
            const auto  status      = cudaGetDeviceCount(&deviceCount);
            if (status != cudaSuccess || deviceCount == 0)
            {
                throw InconsistentInputError(
                        "The NEP/qNEP backend needs a visible CUDA device in mdrun. "
                        "grompp, topology processing and model validation work without a GPU. "
                        "Check the NVIDIA driver, the device visibility (CUDA_VISIBLE_DEVICES) "
                        "and the CUDA architecture of this GROMACS build.");
            }
            model          = createModel(numAtoms);
            modelAtomCount = numAtoms;
        }
        model->N1 = 0;
        model->N2 = numAtoms;
    }

    void resizeModel(int numAtoms)
    {
        if (numAtoms != modelAtomCount || model == nullptr)
        {
            ensureModel(numAtoms);
            qnepGromacs_.reset();
            types.resize(numAtoms);
            positions.resize(3 * numAtoms);
            potential.resize(numAtoms);
            forces.resize(3 * numAtoms);
            virial.resize(9 * numAtoms);
            inputPositions.resize(3 * numAtoms);
            outputForces.resize(3 * numAtoms);
            summary.resize(10);
            hostTypes.resize(numAtoms);
            cachedAtomicNumbers.clear();

            if (hostPositions != nullptr)
            {
                CHECK(cudaFreeHost(hostPositions));
            }
            if (hostForces != nullptr)
            {
                CHECK(cudaFreeHost(hostForces));
            }
            if (hostSummary != nullptr)
            {
                CHECK(cudaFreeHost(hostSummary));
            }
            if (hostCharges != nullptr)
            {
                CHECK(cudaFreeHost(hostCharges));
            }
            CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostPositions),
                                 3 * numAtoms * sizeof(real)));
            CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostForces),
                                 3 * numAtoms * sizeof(real)));
            CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostSummary),
                                 10 * sizeof(double)));
            CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostCharges),
                                 numAtoms * sizeof(float)));
            if (completionEvent == nullptr)
            {
                CHECK(cudaEventCreateWithFlags(&completionEvent, cudaEventDisableTiming));
            }
            modelAtomCount = numAtoms;
        }
        model->N1 = 0;
        model->N2 = numAtoms;
    }

    void resizeOutput(int numAtoms)
    {
        if (numAtoms != outputAtomCount)
        {
            outputForces.resize(3 * numAtoms);
            outputAtomCount = numAtoms;
        }
    }

    std::string modelFile;
    NepModelInfo info;
    NepBackendSettings settings;
    NepBackend backend = NepBackend::Gpumd;
    std::string backendNote;
    bool isQnep = false;
    bool useGromacsCharges = false;
    int chargeMode = 0;
    std::unique_ptr<Potential> model;
    std::unique_ptr<NepQnepGromacs> qnepGromacs_;
    int modelAtomCount = 0;
    int outputAtomCount = 0;
    std::unordered_map<int, int> typeByAtomicNumber;
    GPU_Vector<int> types;
    GPU_Vector<double> positions;
    GPU_Vector<double> potential;
    GPU_Vector<double> forces;
    GPU_Vector<double> virial;
    GPU_Vector<real> inputPositions;
    GPU_Vector<real> outputForces;
    GPU_Vector<int> indexLookup;
    GPU_Vector<double> summary;
    std::vector<int> hostTypes;
    std::vector<int> cachedAtomicNumbers;
    std::vector<int> cachedIndexLookup;
    real* hostPositions = nullptr;
    real* hostForces = nullptr;
    double* hostSummary = nullptr;
    float* hostCharges = nullptr;
    cudaEvent_t completionEvent = nullptr;
    // Stream used by the caller for the following GPU force reduction.
    const DeviceStream* deviceStream = nullptr;
    // Events used to order the default-stream engine call against that stream.
    cudaEvent_t computeStartEvent = nullptr;
    cudaEvent_t computeDoneEvent  = nullptr;
    bool        stepComputeEnergy = true;
    bool        stepComputeVirial = true;
    const MDLogger& logger;
    const MpiComm& mpiComm;
};

NepModel::NepModel(const std::string&        filename,
                   const MDLogger&           logger,
                   const MpiComm&            mpiComm,
                   const bool                gromacsCharges,
                   const NepBackendSettings& settings) :
    impl_(std::make_unique<Impl>(filename, logger, mpiComm, gromacsCharges, settings))
{}

NepModel::~NepModel() = default;

void NepModel::prepareDeviceBuffers(int numModelAtoms, int numOutputAtoms)
{
    impl_->resizeModel(numModelAtoms);
    impl_->resizeOutput(numOutputAtoms);
}

void NepModel::dumpChargesHost(std::vector<float>& out)
{
    const int n = impl_->modelAtomCount;
    if (impl_->model == nullptr || n == 0)
    {
        out.clear();
        return;
    }
    // Both the GPUMD charge models and the WPE engine publish their charges
    // through the Potential interface, so the same accessor works for both.
    GPU_Vector<float>& charges = impl_->model->get_charge_reference();
    out.assign(n, 0.0f);
    if (charges.size() >= static_cast<size_t>(n))
    {
        charges.copy_to_host(out.data());
    }
}

std::vector<int> NepModel::coveredAtomicNumbers() const
{
    std::vector<int> result;
    result.reserve(impl_->typeByAtomicNumber.size());
    for (const auto& entry : impl_->typeByAtomicNumber)
    {
        result.push_back(entry.first);
    }
    return result;
}

bool NepModel::predictChargesHostFrame(ArrayRef<const RVec> x,
                                       const matrix&         box,
                                       int                  homenr,
                                       double               rcoulombNm,
                                       ArrayRef<const int>  atomNumbers,
                                       std::vector<float>&  charges)
{
    (void)homenr;
    (void)homenr;
    if (impl_->backend != NepBackend::Gpumd || !impl_->isQnep || x.empty()
        || atomNumbers.size() != x.size())
    {
        if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
        {
            fprintf(stderr, "[qnep-predict] reject: isQnep=%d x=%zu atomNumbers=%zu\n",
                    static_cast<int>(impl_->isQnep), x.size(), atomNumbers.size());
        }
        return false;
    }
    const int numAtoms = x.size();
    impl_->resizeModel(numAtoms);
    if (impl_->model == nullptr)
    {
        return false;
    }
    for (int i = 0; i < numAtoms; ++i)
    {
        const auto it = impl_->typeByAtomicNumber.find(atomNumbers[i]);
        if (it == impl_->typeByAtomicNumber.end())
        {
            if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
            {
                fprintf(stderr, "[qnep-predict] atom %d has atomic number %d not in model\n",
                        i, atomNumbers[i]);
            }
            return false;
        }
        impl_->hostTypes[i] = it->second;
    }
    impl_->types.copy_from_host(impl_->hostTypes.data());

    if (impl_->qnepGromacs_ == nullptr)
    {
        impl_->qnepGromacs_ = std::make_unique<NepQnepGromacs>(
                static_cast<NEP_Charge*>(impl_->model.get()),
                impl_->chargeMode,
                static_cast<float>(rcoulombNm * c_angstromPerNm),
                numAtoms);
    }

    std::vector<double> posHost(3 * numAtoms);
    for (int i = 0; i < numAtoms; ++i)
    {
        posHost[i]             = static_cast<double>(x[i][XX]) * c_angstromPerNm;
        posHost[numAtoms + i]  = static_cast<double>(x[i][YY]) * c_angstromPerNm;
        posHost[2 * numAtoms + i] = static_cast<double>(x[i][ZZ]) * c_angstromPerNm;
    }
    impl_->positions.copy_from_host(posHost.data());

    Box nepBox;
    std::fill(nepBox.cpu_h, nepBox.cpu_h + 18, 0.0);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            nepBox.cpu_h[3 * j + i] = box[i][j] * c_angstromPerNm;
        }
    }
    nepBox.get_inverse();
    nepBox.set_is_orthogonal();
    nepBox.pbc_x = true;
    nepBox.pbc_y = true;
    nepBox.pbc_z = true;

    charges.resize(numAtoms);
    impl_->qnepGromacs_->predictCharges(
            nepBox, impl_->types, impl_->positions, charges.data(), impl_->potential, numAtoms);
    return true;
}

void NepModel::setDeviceStream(const DeviceStream* stream)
{
    // Debug/benchmark switch: force the legacy host-synchronized path.
    if (std::getenv("GMX_NEP_NO_ASYNC") != nullptr)
    {
        stream = nullptr;
    }
    impl_->deviceStream = stream;
    if (stream != nullptr && impl_->computeStartEvent == nullptr)
    {
        CHECK(cudaEventCreateWithFlags(&impl_->computeStartEvent, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&impl_->computeDoneEvent, cudaEventDisableTiming));
    }
}

void NepModel::setStepComputeFlags(const bool computeEnergy, const bool computeVirial)
{
    impl_->stepComputeEnergy = computeEnergy;
    impl_->stepComputeVirial = computeVirial;
}

void NepModel::waitForDeviceForces() const
{
    if (impl_->completionEvent != nullptr)
    {
        CHECK(cudaEventSynchronize(impl_->completionEvent));
    }
}

DeviceBuffer<RVec> NepModel::deviceForceBuffer() const
{
    if (impl_->outputAtomCount == 0)
    {
        return {};
    }
    return reinterpret_cast<RVec*>(impl_->outputForces.data());
}

bool NepModel::evaluateModelDevice(gmx_enerdata_t*     enerd,
                                   DeviceBuffer<RVec> deviceCoordinates,
                                   GpuEventSynchronizer* deviceCoordinatesReady,
                                   ArrayRef<int>       atomNumbers,
                                   ArrayRef<const int> indexLookup,
                                   int                 numOutputAtoms,
                                   matrix*             box,
                                   PbcType*            pbcType,
                                   matrix*             virial)
{
    if (!deviceCoordinates || box == nullptr || pbcType == nullptr || atomNumbers.empty()
        || indexLookup.size() != atomNumbers.size() || numOutputAtoms <= 0)
    {
        return false;
    }

    const int numAtoms = atomNumbers.size();
    impl_->resizeModel(numAtoms);
    impl_->resizeOutput(numOutputAtoms);

    if (impl_->cachedAtomicNumbers.size() != atomNumbers.size()
        || !std::equal(atomNumbers.begin(),
                       atomNumbers.end(),
                       impl_->cachedAtomicNumbers.begin()))
    {
        for (int i = 0; i < numAtoms; ++i)
        {
            const auto it = impl_->typeByAtomicNumber.find(atomNumbers[i]);
            if (it == impl_->typeByAtomicNumber.end())
            {
                throw InconsistentInputError("Atomic number is not present in the NEP model.");
            }
            impl_->hostTypes[i] = it->second;
        }
        impl_->types.copy_from_host(impl_->hostTypes.data());
        impl_->cachedAtomicNumbers.assign(atomNumbers.begin(), atomNumbers.end());
    }
    if (impl_->cachedIndexLookup.size() != indexLookup.size()
        || !std::equal(indexLookup.begin(),
                       indexLookup.end(),
                       impl_->cachedIndexLookup.begin()))
    {
        for (const int localIndex : indexLookup)
        {
            if (localIndex < 0 || localIndex >= numOutputAtoms)
            {
                throw InconsistentInputError(
                        "Invalid local atom index passed to the NEP device backend.");
            }
        }
        impl_->indexLookup.resize(numAtoms);
        impl_->indexLookup.copy_from_host(indexLookup.data());
        impl_->cachedIndexLookup.assign(indexLookup.begin(), indexLookup.end());
    }

    Box nepBox;
    std::fill(nepBox.cpu_h, nepBox.cpu_h + 18, 0.0);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            nepBox.cpu_h[3 * j + i] = (*box)[i][j] * c_angstromPerNm;
        }
    }
    nepBox.get_inverse();
    nepBox.set_is_orthogonal();
    nepBox.pbc_x = (*pbcType == PbcType::Xyz || *pbcType == PbcType::XY);
    nepBox.pbc_y = (*pbcType == PbcType::Xyz || *pbcType == PbcType::XY);
    nepBox.pbc_z = (*pbcType == PbcType::Xyz);

    // Enqueue our work on the same stream that the following GPU force
    // reduction uses. The engine (GPUMD/WPE) always launches on the legacy
    // default stream, so a pair of events hands that work over to/from our
    // stream without a per-step host synchronization.
    cudaStream_t stream = nullptr;
    if (impl_->deviceStream != nullptr && impl_->deviceStream->isValid())
    {
        stream = impl_->deviceStream->stream();
    }
    const bool async = (stream != nullptr);

    if (deviceCoordinatesReady != nullptr)
    {
        if (async)
        {
            deviceCoordinatesReady->enqueueWaitEvent(*impl_->deviceStream);
        }
        else
        {
            deviceCoordinatesReady->waitForEvent();
        }
    }

    const int atomBlocks =
            (numAtoms + c_postprocessBlockSize - 1) / c_postprocessBlockSize;
    convertPositionsToNep<real, true>
            <<<atomBlocks, c_postprocessBlockSize, 0, stream>>>(numAtoms,
                                                                reinterpret_cast<const real*>(
                                                                        deviceCoordinates),
                                                                impl_->indexLookup.data(),
                                                                impl_->positions.data());
    GPU_CHECK_KERNEL

    CHECK(cudaMemsetAsync(impl_->potential.data(), 0, numAtoms * sizeof(double), stream));
    CHECK(cudaMemsetAsync(impl_->forces.data(), 0, 3 * numAtoms * sizeof(double), stream));
    CHECK(cudaMemsetAsync(impl_->virial.data(), 0, 9 * numAtoms * sizeof(double), stream));
    CHECK(cudaMemsetAsync(impl_->summary.data(), 0, 10 * sizeof(double), stream));

    if (async)
    {
        CHECK(cudaEventRecord(impl_->computeStartEvent, stream));
        CHECK(cudaStreamWaitEvent(nullptr, impl_->computeStartEvent, 0));
    }
    impl_->model->compute(nepBox,
                          impl_->types,
                          impl_->positions,
                          impl_->potential,
                          impl_->forces,
                          impl_->virial);
    if (async)
    {
        CHECK(cudaEventRecord(impl_->computeDoneEvent, nullptr));
        CHECK(cudaStreamWaitEvent(stream, impl_->computeDoneEvent, 0));
    }

    convertForcesAndReduce<real, true>
            <<<atomBlocks, c_postprocessBlockSize, 0, stream>>>(numAtoms,
                                                                impl_->potential.data(),
                                                                impl_->forces.data(),
                                                                impl_->virial.data(),
                                                                impl_->indexLookup.data(),
                                                                impl_->outputForces.data(),
                                                                impl_->summary.data());
    GPU_CHECK_KERNEL

    // The force buffer stays on the device and is consumed by the GPU force
    // reduction on the same stream. Energy/virial are only copied back on the
    // steps that actually need them on the host.
    const bool needSummary = impl_->stepComputeEnergy || impl_->stepComputeVirial;
    if (needSummary || !async)
    {
        CHECK(cudaMemcpyAsync(impl_->hostSummary,
                              impl_->summary.data(),
                              10 * sizeof(double),
                              cudaMemcpyDeviceToHost,
                              stream));
    }
    CHECK(cudaEventRecord(impl_->completionEvent, stream));
    if (needSummary || !async)
    {
        CHECK(cudaEventSynchronize(impl_->completionEvent));
    }

    if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
    {
        fprintf(stderr, "[qnep-debug] output buffer ptr %p atomCount=%d\n",
                static_cast<void*>(impl_->outputForces.data()), impl_->outputAtomCount);
        impl_->outputForces.copy_to_host(impl_->hostForces, 3 * numAtoms);
        double forceSum = 0;
        for (int i = 0; i < 3 * numAtoms; ++i)
        {
            forceSum += std::abs(static_cast<double>(impl_->hostForces[i]));
        }
        fprintf(stderr, "[qnep-debug] mean|output forces| = %g, first: %g %g %g\n",
                forceSum / (3 * numAtoms),
                impl_->hostForces[0],
                impl_->hostForces[1],
                impl_->hostForces[2]);
    }

    if (virial != nullptr && needSummary && impl_->mpiComm.isMainRank())
    {
        const double virialScale = -0.5 * c_evToKjMol;
        for (int component = 0; component < 9; ++component)
        {
            const double sum = impl_->hostSummary[component + 1];
            switch (component)
            {
                case 0: (*virial)[0][0] = virialScale * sum; break;
                case 1: (*virial)[1][1] = virialScale * sum; break;
                case 2: (*virial)[2][2] = virialScale * sum; break;
                case 3: (*virial)[0][1] = virialScale * sum; break;
                case 4: (*virial)[0][2] = virialScale * sum; break;
                case 5: (*virial)[1][2] = virialScale * sum; break;
                case 6: (*virial)[1][0] = virialScale * sum; break;
                case 7: (*virial)[2][0] = virialScale * sum; break;
                case 8: (*virial)[2][1] = virialScale * sum; break;
                default: break;
            }
        }
    }

    enerd->term[InteractionFunction::NeuralNetworkPotentialEnergy] =
            (needSummary && impl_->mpiComm.isMainRank())
                    ? impl_->hostSummary[0] * c_evToKjMol
                    : 0.0;
    return true;
}

void NepModel::evaluateModel(gmx_enerdata_t*                  enerd,
                             ArrayRef<RVec>                   forces,
                             ArrayRef<const int>              indexLookup,
                             ArrayRef<const int>              /* mmIndices */,
                             ArrayRef<const std::string>      /* inputs */,
                             ArrayRef<RVec>                   positions,
                             ArrayRef<int>                    atomNumbers,
                             ArrayRef<int>                    /* atomPairs */,
                             ArrayRef<RVec>                   /* pairShifts */,
                             ArrayRef<RVec>                   /* positionsMM */,
                             ArrayRef<real>                   /* chargesMM */,
                             real                             /* nnpCharge */,
                             ArrayRef<const LinkFrontierAtom> linkFrontier,
                             matrix*                          box,
                             PbcType*                         pbcType,
                             matrix*                          virial)
{
    if (box == nullptr || pbcType == nullptr)
    {
        throw InconsistentInputError("NEP requires a simulation box and PBC information.");
    }

    const int numAtoms = positions.size();
    if (numAtoms == 0 || atomNumbers.size() != numAtoms || indexLookup.size() != numAtoms)
    {
        throw InconsistentInputError("Invalid atom data passed to the NEP backend.");
    }
    impl_->resizeModel(numAtoms);

    if (impl_->cachedAtomicNumbers.size() != atomNumbers.size()
        || !std::equal(atomNumbers.begin(),
                       atomNumbers.end(),
                       impl_->cachedAtomicNumbers.begin()))
    {
        for (int i = 0; i < numAtoms; ++i)
        {
            const auto it = impl_->typeByAtomicNumber.find(atomNumbers[i]);
            if (it == impl_->typeByAtomicNumber.end())
            {
                throw InconsistentInputError("Atomic number is not present in the NEP model.");
            }
            impl_->hostTypes[i] = it->second;
        }
        impl_->types.copy_from_host(impl_->hostTypes.data());
        impl_->cachedAtomicNumbers.assign(atomNumbers.begin(), atomNumbers.end());
    }

    Box nepBox;
    std::fill(nepBox.cpu_h, nepBox.cpu_h + 18, 0.0);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            // GROMACS stores box vectors as matrix rows, whereas GPUMD
            // stores them as columns in cpu_h.
            nepBox.cpu_h[3 * j + i] = (*box)[i][j] * c_angstromPerNm;
        }
    }
    nepBox.get_inverse();
    nepBox.set_is_orthogonal();
    nepBox.pbc_x = (*pbcType == PbcType::Xyz || *pbcType == PbcType::XY);
    nepBox.pbc_y = (*pbcType == PbcType::Xyz || *pbcType == PbcType::XY);
    nepBox.pbc_z = (*pbcType == PbcType::Xyz);

    // RVec storage is contiguous (the same layout used by GROMACS reductions).
    // Stage it in page-locked memory so the transfer can be queued, then
    // transpose and convert nm -> Angstrom on the GPU.
    std::memcpy(impl_->hostPositions,
                positions.data()->as_vec(),
                3 * numAtoms * sizeof(real));
    CHECK(cudaMemcpyAsync(impl_->inputPositions.data(),
                          impl_->hostPositions,
                          3 * numAtoms * sizeof(real),
                          cudaMemcpyHostToDevice));
    const int atomBlocks = (numAtoms + c_postprocessBlockSize - 1)
                           / c_postprocessBlockSize;
    convertPositionsToNep<real, false>
            <<<atomBlocks, c_postprocessBlockSize>>>(numAtoms,
                                                     impl_->inputPositions.data(),
                                                     nullptr,
                                                     impl_->positions.data());
    GPU_CHECK_KERNEL

    // GPUMD force kernels accumulate with +=. The standalone GPUMD force
    // driver clears these arrays before evaluating a potential, so the
    // GROMACS adapter must provide the same initialization.
    CHECK(cudaMemsetAsync(impl_->potential.data(), 0, numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->forces.data(), 0, 3 * numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->virial.data(), 0, 9 * numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->summary.data(), 0, 10 * sizeof(double)));

    impl_->model->compute(nepBox,
                          impl_->types,
                          impl_->positions,
                          impl_->potential,
                          impl_->forces,
                          impl_->virial);

    // Convert the only per-atom output GROMACS needs to its native precision
    // and layout, and reduce energy/virial on the GPU. This avoids copying
    // 13 double arrays to the CPU every MD step.
    impl_->resizeOutput(numAtoms);
    convertForcesAndReduce<real, false>
            <<<atomBlocks, c_postprocessBlockSize>>>(numAtoms,
                                                     impl_->potential.data(),
                                                     impl_->forces.data(),
                                                     impl_->virial.data(),
                                                     nullptr,
                                                     impl_->outputForces.data(),
                                                     impl_->summary.data());
    GPU_CHECK_KERNEL
    CHECK(cudaMemcpyAsync(impl_->hostForces,
                          impl_->outputForces.data(),
                          3 * numAtoms * sizeof(real),
                          cudaMemcpyDeviceToHost));    CHECK(cudaMemcpyAsync(impl_->hostSummary,
                          impl_->summary.data(),
                          10 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(impl_->completionEvent, nullptr));
    CHECK(cudaEventSynchronize(impl_->completionEvent));

    if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
    {
        fprintf(stderr, "[qnep-debug] output buffer ptr %p atomCount=%d\n",
                static_cast<void*>(impl_->outputForces.data()), impl_->outputAtomCount);
        impl_->outputForces.copy_to_host(impl_->hostForces, 3 * numAtoms);
        double forceSum = 0;
        for (int i = 0; i < 3 * numAtoms; ++i)
        {
            forceSum += std::abs(static_cast<double>(impl_->hostForces[i]));
        }
        fprintf(stderr, "[qnep-debug] mean|output forces| = %g, first: %g %g %g\n",
                forceSum / (3 * numAtoms),
                impl_->hostForces[0],
                impl_->hostForces[1],
                impl_->hostForces[2]);
    }

    if (virial != nullptr && impl_->mpiComm.isMainRank())
    {
        // GPUMD stores per-atom virials in the order
        // xx, yy, zz, xy, xz, yz, yx, zx, zy. GROMACS uses the
        // negative force-position virial convention.
        // ForceWithVirial uses the half-virial convention used by the
        // other GROMACS force providers (e.g. PLUMED and Colvars).
        const double virialScale = -0.5 * c_evToKjMol;
        for (int component = 0; component < 9; ++component)
        {
            const double sum = impl_->hostSummary[component + 1];
            switch (component)
            {
                case 0: (*virial)[0][0] = virialScale * sum; break;
                case 1: (*virial)[1][1] = virialScale * sum; break;
                case 2: (*virial)[2][2] = virialScale * sum; break;
                case 3: (*virial)[0][1] = virialScale * sum; break;
                case 4: (*virial)[0][2] = virialScale * sum; break;
                case 5: (*virial)[1][2] = virialScale * sum; break;
                case 6: (*virial)[1][0] = virialScale * sum; break;
                case 7: (*virial)[2][0] = virialScale * sum; break;
                case 8: (*virial)[2][1] = virialScale * sum; break;
                default: break;
            }
        }
    }

    // The NNP input contains link atoms in place of the corresponding MM
    // frontier atoms. Spread their forces back to the two real atoms before
    // mapping model indices to GROMACS local indices.
    if (!linkFrontier.empty())
    {
        t_pbc pbc;
        set_pbc(&pbc, *pbcType, *box);
        for (const auto& link : linkFrontier)
        {
            RVec linkForce;
            const int inputIndexMM = link.getInputIndexMM();
            linkForce[0] =
                    impl_->hostForces[3 * inputIndexMM];
            linkForce[1] =
                    impl_->hostForces[3 * inputIndexMM + 1];
            linkForce[2] =
                    impl_->hostForces[3 * inputIndexMM + 2];
            RVec fNNP;
            RVec fMM;
            std::tie(fNNP, fMM) = link.spreadForce(linkForce, pbc);
            const int inputIndexEmb = link.getInputIndexEmb();
            impl_->hostForces[3 * inputIndexEmb]     = fNNP[0];
            impl_->hostForces[3 * inputIndexEmb + 1] = fNNP[1];
            impl_->hostForces[3 * inputIndexEmb + 2] = fNNP[2];
            impl_->hostForces[3 * inputIndexMM]      = fMM[0];
            impl_->hostForces[3 * inputIndexMM + 1]  = fMM[1];
            impl_->hostForces[3 * inputIndexMM + 2]  = fMM[2];
        }
    }

    for (int i = 0; i < numAtoms; ++i)
    {
        const int localIndex = indexLookup[i];
        if (localIndex >= 0 && localIndex < static_cast<int>(forces.size()))
        {
            forces[localIndex][0] += impl_->hostForces[3 * i];
            forces[localIndex][1] += impl_->hostForces[3 * i + 1];
            forces[localIndex][2] += impl_->hostForces[3 * i + 2];
        }
    }
    enerd->term[InteractionFunction::NeuralNetworkPotentialEnergy] =
            impl_->mpiComm.isMainRank() ? impl_->hostSummary[0] * c_evToKjMol : 0.0;
}

bool NepModel::hasDynamicCharges() const
{
    return impl_->useGromacsCharges && impl_->isQnep
           && impl_->backend == NepBackend::Gpumd;
}

NepBackend NepModel::resolvedBackend() const
{
    return impl_->backend;
}

std::string NepModel::backendDescription() const
{
    std::string description = formatString("%s backend", nepBackendName(impl_->backend));
    if (!impl_->backendNote.empty())
    {
        description += " (" + impl_->backendNote + ")";
    }
    return description;
}


namespace
{

/*! \brief Fills the GPUMD Box structure from the GROMACS box and PBC type. */
void fillNepBox(Box* nepBox, const matrix& box, const PbcType& pbcType)
{
    std::fill(nepBox->cpu_h, nepBox->cpu_h + 18, 0.0);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            nepBox->cpu_h[3 * j + i] = box[i][j] * c_angstromPerNm;
        }
    }
    nepBox->get_inverse();
    nepBox->set_is_orthogonal();
    nepBox->pbc_x = (pbcType == PbcType::Xyz || pbcType == PbcType::XY);
    nepBox->pbc_y = (pbcType == PbcType::Xyz || pbcType == PbcType::XY);
    nepBox->pbc_z = (pbcType == PbcType::Xyz);
}

} // namespace

bool NepModel::updateChargesDevice(ArrayRef<real>             chargeA,
                                   ArrayRef<const int>        atomNumbers,
                                   DeviceBuffer<RVec>         deviceCoordinates,
                                   GpuEventSynchronizer*      deviceCoordinatesReady,
                                   const interaction_const_t& ic,
                                   const PbcType              pbcType,
                                   const matrix&              box,
                                   int                        homenr)
{
    if (impl_->backend != NepBackend::Gpumd || !impl_->useGromacsCharges || !impl_->isQnep
        || !deviceCoordinates || chargeA.empty() || atomNumbers.size() != chargeA.size())
    {
        return false;
    }

    const int numAtoms = chargeA.size();
    impl_->resizeModel(numAtoms);
    impl_->resizeOutput(homenr);

    // Cache the type mapping. The qNEP-GROMACS mode uses the full system with
    // an identity atom order; the shared caching keeps this consistent with
    // the force evaluation phase.
    if (impl_->cachedAtomicNumbers.size() != atomNumbers.size()
        || !std::equal(atomNumbers.begin(),
                       atomNumbers.end(),
                       impl_->cachedAtomicNumbers.begin()))
    {
        for (int i = 0; i < numAtoms; ++i)
        {
            const auto it = impl_->typeByAtomicNumber.find(atomNumbers[i]);
            if (it == impl_->typeByAtomicNumber.end())
            {
                throw InconsistentInputError("Atomic number is not present in the NEP model.");
            }
            impl_->hostTypes[i] = it->second;
        }
        impl_->types.copy_from_host(impl_->hostTypes.data());
        impl_->cachedAtomicNumbers.assign(atomNumbers.begin(), atomNumbers.end());
    }

    if (impl_->qnepGromacs_ == nullptr)
    {
        // The GPUMD neighbor cell list used for the short-range potential
        // needs at least five cells per periodic direction.
        const double minBoxDiagonal = std::min({ box[0][0], box[1][1], box[2][2] });
        // The GPUMD cell list needs at least five cells per periodic
        // direction. Smaller boxes use the per-step neighbor rebuild path.
        if (minBoxDiagonal < 2.5 * ic.coulomb.cutoff)
        {
            throw InconsistentInputError(
                    "qNEP with GROMACS-owned electrostatics requires the box length "
                    "to be at least 2.5 * rcoulomb in every periodic direction "
                    "(GPUMD neighbor cell list requirement).");
        }
        impl_->qnepGromacs_ = std::make_unique<NepQnepGromacs>(
                static_cast<NEP_Charge*>(impl_->model.get()),
                impl_->chargeMode,
                static_cast<float>(ic.coulomb.cutoff * c_angstromPerNm),
                numAtoms);
    }

    const int atomBlocks =
            (numAtoms + c_postprocessBlockSize - 1) / c_postprocessBlockSize;
    convertPositionsToNep<real, false>
            <<<atomBlocks, c_postprocessBlockSize>>>(numAtoms,
                                                     reinterpret_cast<const real*>(
                                                             deviceCoordinates),
                                                     nullptr,
                                                     impl_->positions.data());
    GPU_CHECK_KERNEL

    Box nepBox;
    fillNepBox(&nepBox, box, pbcType);

    // The external GPUMD kernels use the legacy default stream. Wait for the
    // GROMACS coordinate-ready event before launching that work.
    if (deviceCoordinatesReady != nullptr)
    {
        deviceCoordinatesReady->waitForEvent();
    }

    CHECK(cudaMemsetAsync(impl_->potential.data(), 0, numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->forces.data(), 0, 3 * numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->virial.data(), 0, 9 * numAtoms * sizeof(double)));

    impl_->qnepGromacs_->predictCharges(nepBox,
                                        impl_->types,
                                        impl_->positions,
                                        impl_->hostCharges,
                                        impl_->potential,
                                        numAtoms);

    for (int i = 0; i < numAtoms; ++i)
    {
        if (std::getenv("GMX_QNEP_FIXED_CHARGES") != nullptr)
        {
            chargeA[i] = (i % 2 == 0) ? real(0.4) : real(-0.4);
        }
        else
        {
            chargeA[i] = static_cast<real>(impl_->hostCharges[i]);
        }
    }
    return true;
}

bool NepModel::evaluateModelDeviceQnepGromacs(gmx_enerdata_t*           enerd,
                                              DeviceBuffer<float>       coulombPotential,
                                              GpuEventSynchronizer*     coulombPotentialReady,
                                              ArrayRef<int>             atomNumbers,
                                              ArrayRef<const int>       indexLookup,
                                              int                       numOutputAtoms,
                                              const interaction_const_t& ic,
                                              matrix*                   box,
                                              PbcType*                  pbcType,
                                              matrix*                   virial,
                                              bool                      computeEnergy,
                                              bool                      computeVirial)
{
    if (impl_->backend != NepBackend::Gpumd || !impl_->useGromacsCharges || !impl_->isQnep
        || !coulombPotential || box == nullptr || pbcType == nullptr || atomNumbers.empty()
        || indexLookup.size() != atomNumbers.size() || numOutputAtoms <= 0)
    {
        return false;
    }

    const int numAtoms = atomNumbers.size();
    impl_->resizeModel(numAtoms);
    impl_->resizeOutput(numOutputAtoms);

    // Cache the type mapping and the index lookup.
    if (impl_->cachedAtomicNumbers.size() != atomNumbers.size()
        || !std::equal(atomNumbers.begin(),
                       atomNumbers.end(),
                       impl_->cachedAtomicNumbers.begin()))
    {
        for (int i = 0; i < numAtoms; ++i)
        {
            const auto it = impl_->typeByAtomicNumber.find(atomNumbers[i]);
            if (it == impl_->typeByAtomicNumber.end())
            {
                throw InconsistentInputError("Atomic number is not present in the NEP model.");
            }
            impl_->hostTypes[i] = it->second;
        }
        impl_->types.copy_from_host(impl_->hostTypes.data());
        impl_->cachedAtomicNumbers.assign(atomNumbers.begin(), atomNumbers.end());
    }
    if (impl_->cachedIndexLookup.size() != indexLookup.size()
        || !std::equal(indexLookup.begin(),
                       indexLookup.end(),
                       impl_->cachedIndexLookup.begin()))
    {
        for (const int localIndex : indexLookup)
        {
            if (localIndex < 0 || localIndex >= numOutputAtoms)
            {
                throw InconsistentInputError(
                        "Invalid local atom index passed to the NEP device backend.");
            }
        }
        impl_->indexLookup.resize(numAtoms);
        impl_->indexLookup.copy_from_host(indexLookup.data());
        impl_->cachedIndexLookup.assign(indexLookup.begin(), indexLookup.end());
    }

    // Wait for the GROMACS PME gather (which wrote the per-atom potential).
    if (coulombPotentialReady != nullptr)
    {
        coulombPotentialReady->waitForEvent();
    }

    Box nepBox;
    fillNepBox(&nepBox, *box, *pbcType);

    impl_->qnepGromacs_->computeForces(nepBox,
                                       impl_->types,
                                       impl_->positions,
                                       reinterpret_cast<const float*>(coulombPotential),
                                       static_cast<float>(ic.coulomb.ewaldCoeff),
                                       static_cast<float>(ic.coulomb.ewaldShift),
                                       static_cast<float>(ic.coulomb.epsfac),
                                       static_cast<float>(ic.coulomb.cutoff),
                                       impl_->potential,
                                       impl_->forces,
                                       impl_->virial,
                                       numAtoms);

    if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
    {
        std::vector<double> debugForces(3 * numAtoms);
        static_cast<NEP_Charge*>(impl_->model.get())->nep_data.D_real.copy_to_host(
                reinterpret_cast<float*>(impl_->hostForces), numAtoms);
        double sum = 0;
        for (int i = 0; i < numAtoms; ++i)
        {
            sum += std::abs(impl_->hostForces[i]);
        }
        fprintf(stderr, "[qnep-debug] mean|D_real| = %g (eV/e), first: %g %g %g\n",
                sum / numAtoms,
                impl_->hostForces[0],
                impl_->hostForces[1],
                impl_->hostForces[2]);
        impl_->forces.copy_to_host(debugForces.data());
        double forceSum = 0;
        for (int i = 0; i < 3 * numAtoms; ++i)
        {
            forceSum += std::abs(debugForces[i]);
        }
        fprintf(stderr, "[qnep-debug] mean|nep forces (eV/A)| = %g, first: %g %g %g\n",
                forceSum / (3 * numAtoms),
                debugForces[0],
                debugForces[1],
                debugForces[2]);
    }

    const int atomBlocks =
            (numAtoms + c_postprocessBlockSize - 1) / c_postprocessBlockSize;
    CHECK(cudaMemsetAsync(impl_->summary.data(), 0, 10 * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->outputForces.data(),
                          0,
                          3 * numOutputAtoms * sizeof(real)));
    convertForcesAndReduce<real, true>
            <<<atomBlocks, c_postprocessBlockSize>>>(numAtoms,
                                                     impl_->potential.data(),
                                                     impl_->forces.data(),
                                                     impl_->virial.data(),
                                                     impl_->indexLookup.data(),
                                                     impl_->outputForces.data(),
                                                     impl_->summary.data());
    GPU_CHECK_KERNEL

    // The device forces are consumed by the GPU force reduction (or by the
    // host-side D2H on the alternating wait path) without a host
    // synchronization. The energy/virial summary only needs to reach the host
    // on steps where GROMACS actually uses it.
    if (computeEnergy || computeVirial)
    {
        CHECK(cudaMemcpyAsync(impl_->hostSummary,
                              impl_->summary.data(),
                              10 * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(impl_->completionEvent, nullptr));
        CHECK(cudaEventSynchronize(impl_->completionEvent));
    }

    if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
    {
        fprintf(stderr, "[qnep-debug] output buffer ptr %p atomCount=%d\n",
                static_cast<void*>(impl_->outputForces.data()), impl_->outputAtomCount);
        impl_->outputForces.copy_to_host(impl_->hostForces, 3 * numAtoms);
        double forceSum = 0;
        for (int i = 0; i < 3 * numAtoms; ++i)
        {
            forceSum += std::abs(static_cast<double>(impl_->hostForces[i]));
        }
        fprintf(stderr, "[qnep-debug] mean|output forces| = %g, first: %g %g %g\n",
                forceSum / (3 * numAtoms),
                impl_->hostForces[0],
                impl_->hostForces[1],
                impl_->hostForces[2]);
    }

    if (computeEnergy || computeVirial)
    {
        if (virial != nullptr && impl_->mpiComm.isMainRank())
        {
            const double virialScale = -0.5 * c_evToKjMol;
            for (int component = 0; component < 9; ++component)
            {
                const double sum = impl_->hostSummary[component + 1];
                switch (component)
                {
                    case 0: (*virial)[0][0] = virialScale * sum; break;
                    case 1: (*virial)[1][1] = virialScale * sum; break;
                    case 2: (*virial)[2][2] = virialScale * sum; break;
                    case 3: (*virial)[0][1] = virialScale * sum; break;
                    case 4: (*virial)[0][2] = virialScale * sum; break;
                    case 5: (*virial)[1][2] = virialScale * sum; break;
                    case 6: (*virial)[1][0] = virialScale * sum; break;
                    case 7: (*virial)[2][0] = virialScale * sum; break;
                    case 8: (*virial)[2][1] = virialScale * sum; break;
                    default: break;
                }
            }
        }

        // Only the NEP part is reported here; the Coulomb energy terms are
        // produced by the GROMACS non-bonded/PME calculation itself.
        enerd->term[InteractionFunction::NeuralNetworkPotentialEnergy] =
                impl_->mpiComm.isMainRank() ? impl_->hostSummary[0] * c_evToKjMol : 0.0;
    }
    return true;
}

} // namespace gmx
