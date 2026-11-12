/*
 * GROMACS NEP backend.
 *
 * This file is part of a local GROMACS/GPUMD integration. The NEP evaluator
 * itself is derived from GPUMD and remains under the GPUMD license.
 */
#ifndef GMX_APPLIED_FORCES_NEPMODEL_H
#define GMX_APPLIED_FORCES_NEPMODEL_H

#include <memory>
#include <string>
#include <vector>

#include "gromacs/applied_forces/nnpot/nepbackendsettings.h"
#include "gromacs/applied_forces/nnpot/nepmodelinfo.h"
#include "gromacs/applied_forces/nnpot/nnpotmodel.h"

struct interaction_const_t;

class DeviceStream;

namespace gmx
{

class MDLogger;
class MpiComm;

/*! \brief Which engine evaluates the NEP/qNEP model. */
// (NepBackend, nepBackendName(), nepBackendFromString() and
// NepBackendSettings live in nepbackendsettings.h so grompp can use them
// without CUDA.)

class NepModel final : public INNPotModel
{
public:
    NepModel(const std::string&        filename,
             const MDLogger&           logger,
             const MpiComm&            mpiComm,
             bool                      gromacsCharges = false,
             const NepBackendSettings& settings       = NepBackendSettings());
    ~NepModel() override;

    void evaluateModel(gmx_enerdata_t*                  enerd,
                       ArrayRef<RVec>                   forces,
                       ArrayRef<const int>              indexLookup,
                       ArrayRef<const int>              mmIndices,
                       ArrayRef<const std::string>      inputs,
                       ArrayRef<RVec>                   positions,
                       ArrayRef<int>                    atomNumbers,
                       ArrayRef<int>                    atomPairs,
                       ArrayRef<RVec>                   pairShifts,
                       ArrayRef<RVec>                   positionsMM,
                       ArrayRef<real>                   chargesMM,
                       real                             nnpCharge,
                       ArrayRef<const LinkFrontierAtom> linkFrontier,
                       matrix*                          box,
                       PbcType*                         pbcType,
                       matrix*                          virial = nullptr) override;

    bool outputsForces() const override { return true; }

    bool evaluateModelDevice(gmx_enerdata_t*,
                             DeviceBuffer<RVec>,
                             GpuEventSynchronizer*,
                             ArrayRef<int>,
                             ArrayRef<const int>,
                             int,
                             matrix*,
                             PbcType*,
                             matrix*) override;

    DeviceBuffer<RVec> deviceForceBuffer() const override;

    void setDeviceStream(const DeviceStream* stream) override;
    void setStepComputeFlags(bool computeEnergy, bool computeVirial) override;
    void waitForDeviceForces() const override;


    void prepareDeviceBuffers(int numModelAtoms, int numOutputAtoms);

    //! Copy the current qNEP model charges (model-atom order) to the host.
    void dumpChargesHost(std::vector<float>& out);

    //! Atomic numbers covered by the model (the qNEP region elements).
    std::vector<int> coveredAtomicNumbers() const;

    //! Predict the qNEP dynamic charges for a host coordinate frame.
    //! Requires a live CUDA context and the GPUMD backend.
    bool predictChargesHostFrame(ArrayRef<const RVec> x,
                                 const matrix&         box,
                                 int                   homenr,
                                 double                rcoulombNm,
                                 ArrayRef<const int>   atomNumbers,
                                 std::vector<float>&   charges);

    //! Whether this model provides dynamic charges handled by GROMACS
    //! electrostatics (qNEP with nnpot-gromacs-charges, GPUMD backend only).
    bool hasDynamicCharges() const;

    //! Phase 1: predict neutralized dynamic charges from device coordinates.
    bool updateChargesDevice(ArrayRef<real>             chargeA,
                             ArrayRef<const int>        atomNumbers,
                             DeviceBuffer<RVec>         deviceCoordinates,
                             GpuEventSynchronizer*      deviceCoordinatesReady,
                             const interaction_const_t& ic,
                             PbcType                    pbcType,
                             const matrix&              box,
                             int                        homenr);

    //! Phase 2: NEP short-range energy/forces/virial plus chain-rule forces
    //! from the GROMACS per-atom Coulomb potential (GPUMD backend only).
    bool evaluateModelDeviceQnepGromacs(gmx_enerdata_t*            enerd,
                                        DeviceBuffer<float>        coulombPotential,
                                        GpuEventSynchronizer*      coulombPotentialReady,
                                        ArrayRef<int>              atomNumbers,
                                        ArrayRef<const int>        indexLookup,
                                        int                        numOutputAtoms,
                                        const interaction_const_t& ic,
                                        matrix*                    box,
                                        PbcType*                   pbcType,
                                        matrix*                    virial,
                                        bool                       computeEnergy = true,
                                        bool                       computeVirial = true);

    //! Human readable backend selection, for log output.
    std::string backendDescription() const;

    //! The backend actually used (Auto is resolved in the constructor).
    NepBackend resolvedBackend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
