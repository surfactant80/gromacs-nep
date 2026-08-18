/*
 * GROMACS qNEP adapter for GROMACS-owned electrostatics.
 *
 * This file is part of a local GROMACS/GPUMD integration. It reuses the
 * unmodified GPUMD NEP_Charge parser and data structures, but replaces the
 * GPUMD PPPM/Ewald electrostatics with GROMACS non-bonded + PME:
 *
 *   - Phase 1 predicts the neutralized dynamic charges and stores the NEP
 *     descriptor data (Fp, charge_derivative, sum_fxyz) on the device.
 *   - Phase 2 computes the NEP short-range forces and the charge-response
 *     chain-rule forces (dE/dq)(dq/dR) using the per-atom Coulomb potential
 *     that GROMACS computed during the non-bonded/PME force evaluation.
 *
 * Only the adapter code in this directory links against the GPUMD internals;
 * the vendored tree at src/external/gpumd remains unmodified.
 */
#ifndef GMX_APPLIED_FORCES_NEPMODEL_QNEP_GROMACS_H
#define GMX_APPLIED_FORCES_NEPMODEL_QNEP_GROMACS_H

#include <memory>

#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"

class NEP_Charge;

namespace gmx
{

/*! \brief GPU adapter that evaluates a qNEP model without its native PPPM/Ewald
 * electrostatics, so GROMACS non-bonded + PME can own the electrostatics.
 *
 * Phase 1 predicts neutralized dynamic charges and stores all descriptor data;
 * phase 2 computes the NEP short-range forces plus the charge-response
 * chain-rule forces from the GROMACS per-atom Coulomb potential. Both phases
 * run on the legacy CUDA default stream and synchronize at the end.
 *
 * Charge mode 1 (full real + reciprocal electrostatics): the chain rule uses
 * the total GROMACS Coulomb potential. Charge mode 2 (reciprocal space only):
 * the GPUMD-style real-space part (alpha = pi/rc_radial) is subtracted from
 * the energy, forces, virial and the chain-rule potential.
 */
class NepQnepGromacs
{
public:
    /*! \brief Constructor.
     *
     * \param model       An initialized NEP_Charge model (owns all parameters).
     * \param chargeMode  qNEP charge mode (1 or 2).
     * \param rcCoulombA  GROMACS real-space Coulomb cutoff in Angstrom, used to
     *                    build the per-atom short-range potential neighbor list.
     * \param numAtoms    Number of model atoms.
     */
    NepQnepGromacs(NEP_Charge* model, int chargeMode, float rcCoulombA, int numAtoms);
    ~NepQnepGromacs();

    /*! \brief Phase 1: evaluate descriptors and charges, enforce charge
     * neutrality and copy the charges to the host.
     *
     * The caller must have filled \p positions with the current coordinates
     * (Angstrom, double) and converted \p types, and must have zeroed the
     * potential/force/virial buffers. On return, \p hostCharges contains the
     * neutralized charges, the short-range NEP energy is accumulated in
     * \p potential and the descriptor data is stored on the device for the
     * subsequent computeForces() call.
     */
    void predictCharges(Box&                      box,
                        const GPU_Vector<int>&    types,
                        const GPU_Vector<double>& positions,
                        float*                    hostCharges,
                        GPU_Vector<double>&       potential,
                        int                       numAtoms);

    /*! \brief Phase 2: compute the NEP energy, forces and virial including the
     * charge-response chain-rule forces from the GROMACS per-atom potential.
     *
     * \param d_pmePotential  Per-atom reciprocal-space Coulomb potential from
     *                        GROMACS PME (device, kJ mol^-1 e^-1, size numAtoms).
     * \param betaNm          GROMACS ewald coefficient (1/nm).
     * \param ewaldShiftNm    GROMACS ewald shift (1/nm).
     * \param epsfac          GROMACS electrostatics prefactor.
     * \param rcCoulombNm     GROMACS Coulomb cutoff (nm).
     */
    void computeForces(Box&                      box,
                       const GPU_Vector<int>&    types,
                       const GPU_Vector<double>& positions,
                       const float*              d_pmePotential,
                       float                     betaNm,
                       float                     ewaldShiftNm,
                       float                     epsfac,
                       float                     rcCoulombNm,
                       GPU_Vector<double>&       potential,
                       GPU_Vector<double>&       forces,
                       GPU_Vector<double>&       virial,
                       int                       numAtoms);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
