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

#include "gromacs/applied_forces/nnpot/nnpotmodel.h"

struct interaction_const_t;

namespace gmx
{

class MDLogger;
class MpiComm;

class NepModel final : public INNPotModel
{
public:
    NepModel(const std::string& filename,
             const MDLogger&    logger,
             const MpiComm&     mpiComm,
             bool               gromacsCharges = false);
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

    void prepareDeviceBuffers(int numModelAtoms, int numOutputAtoms);

    //! Whether this model provides dynamic charges that are handled by
    //! GROMACS electrostatics (qNEP with nnpot-gromacs-charges).
    bool hasDynamicCharges() const;

    //! Phase 1: predict neutralized dynamic charges from the device
    //! coordinates and write them into \p chargeA (host).
    bool updateChargesDevice(ArrayRef<real>             chargeA,
                             ArrayRef<const int>        atomNumbers,
                             DeviceBuffer<RVec>         deviceCoordinates,
                             GpuEventSynchronizer*      deviceCoordinatesReady,
                             const interaction_const_t& ic,
                             PbcType                     pbcType,
                             const matrix&               box,
                             int                        homenr);

    //! Phase 2: compute the NEP short-range energy/forces/virial and the
    //! charge-response chain-rule forces using the GROMACS per-atom Coulomb
    //! potential, directly in the device force buffer. Must be called after
    //! updateChargesDevice() with the same atom data.
    bool evaluateModelDeviceQnepGromacs(gmx_enerdata_t*           enerd,
                                        DeviceBuffer<float>       coulombPotential,
                                        GpuEventSynchronizer*     coulombPotentialReady,
                                        ArrayRef<int>             atomNumbers,
                                        ArrayRef<const int>       indexLookup,
                                        int                       numOutputAtoms,
                                        const interaction_const_t& ic,
                                        matrix*                   box,
                                        PbcType*                  pbcType,
                                        matrix*                   virial,
                                        bool                      computeEnergy = true,
                                        bool                      computeVirial = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
