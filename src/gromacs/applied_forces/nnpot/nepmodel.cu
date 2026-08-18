#include "gmxpre.h"

#include "nepmodel.h"

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

#include "nepmodel_qnep_gromacs.h"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include "force/nep.cuh"
#include "force/nep_charge.cuh"

namespace
{

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

std::vector<std::string> firstLineTokens(const std::string& filename)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw std::runtime_error("Cannot open NEP model file: " + filename);
    }
    std::string line;
    while (std::getline(input, line))
    {
        if (line.find_first_not_of(" \t\r\n") != std::string::npos)
        {
            std::istringstream stream(line);
            std::vector<std::string> result;
            std::string token;
            while (stream >> token)
            {
                result.push_back(token);
            }
            return result;
        }
    }
    throw std::runtime_error("NEP model file is empty: " + filename);
}

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
#ifdef _WIN32
    char buffer[_MAX_PATH];
    if (_fullpath(buffer, filename.c_str(), _MAX_PATH) == nullptr)
    {
        throw std::runtime_error("Could not resolve NEP model path: " + filename);
    }
    return buffer;
#else
    char* resolved = realpath(filename.c_str(), nullptr);
    if (resolved == nullptr)
    {
        throw std::runtime_error("Could not resolve NEP model path: " + filename);
    }
    std::string result(resolved);
    std::free(resolved);
    return result;
#endif
}

bool isQnepHeader(const std::string& header)
{
    return header == "nep4_charge1" || header == "nep4_zbl_charge1"
           || header == "nep4_charge2" || header == "nep4_zbl_charge2";
}

/*! \brief CPU-side structural validation of a GPUMD NEP model file.
 *
 * Replicates the arithmetic of the GPUMD parser to verify that the file
 * contains all expected descriptor/ANN parameter values. This allows grompp
 * to reject a broken model file without initializing CUDA.
 */
void validateNepFileStructure(const std::string& filename)
{
    using gmx::InconsistentInputError;
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

    // Skip leading blank lines and read the header tokens.
    size_t lineIndex = 0;
    while (lineIndex < lines.size() && lines[lineIndex].find_first_not_of(" \t\r\n") == std::string::npos)
    {
        ++lineIndex;
    }
    auto tokensAt = [&](size_t index) -> std::vector<std::string>
    {
        if (index >= lines.size())
        {
            throw InconsistentInputError("NEP model file is truncated: " + filename);
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
            throw InconsistentInputError("NEP model file has an empty parameter line: " + filename);
        }
        return result;
    };

    const std::vector<std::string> header = tokensAt(lineIndex++);
    if (header.size() < 3)
    {
        throw InconsistentInputError("Invalid NEP model header.");
    }
    const bool hasZbl = header[0].find("zbl") != std::string::npos;
    const int  numTypes = std::stoi(header[1]);
    if (numTypes <= 0 || static_cast<int>(header.size()) != numTypes + 2)
    {
        throw InconsistentInputError("Invalid NEP model first line.");
    }

    if (hasZbl)
    {
        const auto zblTokens = tokensAt(lineIndex++);
        if (zblTokens.size() < 3 || zblTokens.size() > 4 || zblTokens[0] != "zbl")
        {
            throw InconsistentInputError("Invalid ZBL line in NEP model.");
        }
        const double rcInner = std::stod(zblTokens[1]);
        const double rcOuter = std::stod(zblTokens[2]);
        if (rcInner == 0 && rcOuter == 0)
        {
            // Flexible ZBL: an extra 10 * ntypes*(ntypes+1)/2 values follow
            // the regular parameter stream.
        }
    }
    const auto cutoffTokens = tokensAt(lineIndex++);
    if (cutoffTokens.size() != 5 || cutoffTokens[0] != "cutoff")
    {
        throw InconsistentInputError("Invalid cutoff line in NEP model.");
    }
    const auto nMaxTokens = tokensAt(lineIndex++);
    if (nMaxTokens.size() != 3 || nMaxTokens[0] != "n_max")
    {
        throw InconsistentInputError("Invalid n_max line in NEP model.");
    }
    const int nMaxRadial  = std::stoi(nMaxTokens[1]);
    const int nMaxAngular = std::stoi(nMaxTokens[2]);
    const auto basisTokens = tokensAt(lineIndex++);
    if (basisTokens.size() != 3 || basisTokens[0] != "basis_size")
    {
        throw InconsistentInputError("Invalid basis_size line in NEP model.");
    }
    const int basisRadial  = std::stoi(basisTokens[1]);
    const int basisAngular = std::stoi(basisTokens[2]);
    const auto lMaxTokens = tokensAt(lineIndex++);
    if (lMaxTokens.size() < 4 || lMaxTokens[0] != "l_max")
    {
        throw InconsistentInputError("Invalid l_max line in NEP model.");
    }
    const int lMax = std::stoi(lMaxTokens[1]);
    int       numL = lMax;
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
        throw InconsistentInputError("Invalid ANN line in NEP model.");
    }
    const int neurons = std::stoi(annTokens[1]);

    const int dim        = (nMaxRadial + 1) + (nMaxAngular + 1) * numL;
    const int numTypesSq = numTypes * numTypes;
    // Charge1/charge2 models store (dim+3) ANN parameters per neuron per type
    // (w0, b0, w1, c) plus two shared parameters. Plain NEP4 models store
    // (dim+2) ANN parameters per neuron per type (w0, b0, w1) plus one shared
    // parameter. Both store dim q_scaler values after the descriptors.
    const bool isQnep      = isQnepHeader(header[0]);
    const int  numParaAnn  = isQnep ? (dim + 3) * neurons * numTypes + 2
                                    : (dim + 2) * neurons * numTypes + 1;
    const int  numParaDescr = numTypesSq
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
                "NEP model file is truncated: expected at least " + std::to_string(expectedValues)
                + " parameter values, found " + std::to_string(valueCount) + ".");
     }
}

// State for the run.in isolation. The GPUMD constructors terminate the
// process with exit() on errors (CUDA failures, input errors), which bypasses
// C++ unwinding. The atexit handler removes the temporary blank run.in and
// restores a displaced original run.in in that case.
std::mutex  g_runInIsolationMutex;
bool        g_runInIsolated = false;
std::string g_runInBackup;


void restoreRunInAfterExit()
{
    if (g_runInIsolated)
    {
        std::remove("run.in");
        g_runInIsolated = false;
        if (!g_runInBackup.empty())
        {
            std::rename(g_runInBackup.c_str(), "run.in");
            g_runInBackup.clear();
        }
    }
}

std::unique_ptr<Potential> makeNepModel(const std::string& filename,
                                        const std::string& modelHeader,
                                        int                numAtoms)
{
    // Current GPUMD initializes its optional DFT-D3 add-on by reading a
    // standalone-GPUMD run.in file in NEP's constructor. GROMACS must provide
    // an empty one and must not accidentally consume an unrelated file in the
    // run directory. Serialize construction, temporarily move an existing
    // run.in aside, and restore it afterward. Avoid changing the process
    // working directory: thread-MPI ranks share it.
    std::lock_guard<std::mutex> lock(g_runInIsolationMutex);

    const std::string absoluteModelPath = absolutePath(filename);
    std::random_device randomDevice;
    const std::string runInput = "run.in";
    std::ifstream     existingRunInput(runInput);
    const bool        hadRunInput = existingRunInput.good();
    existingRunInput.close();
    std::string       backupRunInput;
    if (hadRunInput)
    {
        for (int attempt = 0; attempt < 100; ++attempt)
        {
            backupRunInput = "run.in.gromacs-nep-" + std::to_string(randomDevice())
                             + "-" + std::to_string(randomDevice());
            std::ifstream existingBackup(backupRunInput);
            const bool backupExists = existingBackup.good();
            existingBackup.close();
            if (!backupExists && std::rename(runInput.c_str(), backupRunInput.c_str()) == 0)
            {
                break;
            }
            backupRunInput.clear();
        }
        if (backupRunInput.empty())
        {
            throw std::runtime_error("Could not temporarily isolate run.in for GPUMD NEP.");
        }
    }

    static bool atexitRegistered = []()
    {
        std::atexit(restoreRunInAfterExit);
        return true;
    }();
    GMX_UNUSED_VALUE(atexitRegistered);
    g_runInIsolated = true;
    g_runInBackup   = backupRunInput;

    try
    {
        std::ofstream(runInput).close();
        std::unique_ptr<Potential> model;
        if (isQnepHeader(modelHeader))
        {
            model = std::make_unique<NEP_Charge>(absoluteModelPath.c_str(), numAtoms);
        }
        else
        {
            model = std::make_unique<NEP>(absoluteModelPath.c_str(), numAtoms);
        }
        std::remove(runInput.c_str());
        if (hadRunInput
            && std::rename(backupRunInput.c_str(), runInput.c_str()) != 0)
        {
            throw std::runtime_error("Could not restore run.in after GPUMD NEP initialization.");
        }
        g_runInIsolated = false;
        g_runInBackup.clear();
        return model;
    }
    catch (...)
    {
        g_runInIsolated = false;
        g_runInBackup.clear();
        std::remove(runInput.c_str());
        if (hadRunInput)
        {
            std::rename(backupRunInput.c_str(), runInput.c_str());
        }
        throw;
    }
}

} // namespace

namespace gmx
{

struct NepModel::Impl
{
    explicit Impl(const std::string& filename,
                  const MDLogger&    logger,
                  const MpiComm&     mpiComm,
                  const bool         gromacsCharges) :
        modelFile(absolutePath(filename)),
        logger(logger),
        mpiComm(mpiComm)
    {
        const auto tokens = firstLineTokens(modelFile);
        if (tokens.size() < 3
            || (tokens[0] != "nep4" && tokens[0] != "nep4_zbl"
                && tokens[0] != "nep5" && tokens[0] != "nep5_zbl"
                && !isQnepHeader(tokens[0])))
        {
            throw InconsistentInputError(
                    "GROMACS NEP backend supports NEP4/NEP5 and qNEP charge1/charge2 "
                    "potential files.");
        }
        modelHeader = tokens[0];
        isQnep = isQnepHeader(modelHeader);
        if (modelHeader == "nep4_charge1" || modelHeader == "nep4_zbl_charge1")
        {
            chargeMode = 1;
        }
        else if (modelHeader == "nep4_charge2" || modelHeader == "nep4_zbl_charge2")
        {
            chargeMode = 2;
        }
        if (gromacsCharges)
        {
            if (!isQnep)
            {
                throw InconsistentInputError(
                        "nnpot-gromacs-charges requires a qNEP charge1/charge2 model.");
            }
            if (modelHeader.find("zbl") != std::string::npos)
            {
                throw InconsistentInputError(
                        "GROMACS-owned qNEP electrostatics does not support ZBL qNEP "
                        "models yet.");
            }
        }
        useGromacsCharges = gromacsCharges;

        const int numTypes = std::stoi(tokens[1]);
        if (numTypes <= 0 || static_cast<int>(tokens.size()) != numTypes + 2)
        {
            throw InconsistentInputError("Invalid NEP model first line.");
        }

        typeByAtomicNumber.reserve(numTypes);
        for (int i = 0; i < numTypes; ++i)
        {
            const int z = atomicNumber(tokens[i + 2]);
            if (z < 0)
            {
                throw InconsistentInputError("Unknown element in NEP model: " + tokens[i + 2]);
            }
            typeByAtomicNumber.emplace(z, i);
        }
        // Validate the model file structure on the CPU so that grompp does not
        // need a GPU. The actual GPUMD model (and its CUDA buffers) is created
        // lazily on the first force evaluation.
        validateNepFileStructure(modelFile);
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

    //! Create the GPUMD model (and its CUDA buffers) with the given number
    //! of atoms, unless it already exists with the right size.
    void ensureModel(int numAtoms)
    {
        if (model == nullptr || modelAtomCount != numAtoms)
        {
            model = makeNepModel(modelFile, modelHeader, numAtoms);
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
    std::string modelHeader;
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
    const MDLogger& logger;
    const MpiComm& mpiComm;
};

NepModel::NepModel(const std::string& filename,
                   const MDLogger&    logger,
                   const MpiComm&     mpiComm,
                   const bool         gromacsCharges) :
    impl_(std::make_unique<Impl>(filename, logger, mpiComm, gromacsCharges))
{}

NepModel::~NepModel() = default;

void NepModel::prepareDeviceBuffers(int numModelAtoms, int numOutputAtoms)
{
    impl_->resizeModel(numModelAtoms);
    impl_->resizeOutput(numOutputAtoms);
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

    // The external GPUMD kernels use the legacy default stream. Consume the
    // GROMACS coordinate-ready event on the host before launching that work.
    // This still avoids both the coordinate D2H and NEP-force H2D transfers.
    if (deviceCoordinatesReady != nullptr)
    {
        deviceCoordinatesReady->waitForEvent();
    }

    const int atomBlocks =
            (numAtoms + c_postprocessBlockSize - 1) / c_postprocessBlockSize;
    convertPositionsToNep<real, true>
            <<<atomBlocks, c_postprocessBlockSize>>>(numAtoms,
                                                     reinterpret_cast<const real*>(
                                                             deviceCoordinates),
                                                     impl_->indexLookup.data(),
                                                     impl_->positions.data());
    GPU_CHECK_KERNEL

    CHECK(cudaMemsetAsync(impl_->potential.data(), 0, numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->forces.data(), 0, 3 * numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->virial.data(), 0, 9 * numAtoms * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->summary.data(), 0, 10 * sizeof(double)));
    CHECK(cudaMemsetAsync(impl_->outputForces.data(),
                          0,
                          3 * numOutputAtoms * sizeof(real)));

    impl_->model->compute(nepBox,
                          impl_->types,
                          impl_->positions,
                          impl_->potential,
                          impl_->forces,
                          impl_->virial);

    convertForcesAndReduce<real, true>
            <<<atomBlocks, c_postprocessBlockSize>>>(numAtoms,
                                                     impl_->potential.data(),
                                                     impl_->forces.data(),
                                                     impl_->virial.data(),
                                                     impl_->indexLookup.data(),
                                                     impl_->outputForces.data(),
                                                     impl_->summary.data());
    GPU_CHECK_KERNEL
    CHECK(cudaMemcpyAsync(impl_->hostSummary,
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
            impl_->mpiComm.isMainRank() ? impl_->hostSummary[0] * c_evToKjMol : 0.0;
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
                          cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpyAsync(impl_->hostSummary,
                          impl_->summary.data(),
                          10 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    // Wait only for the NEP adapter's output event. Synchronizing the legacy
    // default stream also waits for unrelated GROMACS GPU work.
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
    return impl_->useGromacsCharges && impl_->isQnep;
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
    if (!impl_->useGromacsCharges || !impl_->isQnep || !deviceCoordinates || chargeA.empty()
        || atomNumbers.size() != chargeA.size())
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
    if (!impl_->useGromacsCharges || !impl_->isQnep || !coulombPotential || box == nullptr
        || pbcType == nullptr || atomNumbers.empty() || indexLookup.size() != atomNumbers.size()
        || numOutputAtoms <= 0)
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
