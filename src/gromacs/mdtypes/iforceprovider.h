/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2016- The GROMACS Authors
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
/*! \libinternal \file
 * \brief
 * Declares gmx::IForceProvider and ForceProviders.
 *
 * See \ref page_mdmodules for an overview of this and associated interfaces.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \author Carsten Kutzner <ckutzne@gwdg.de>
 * \inlibraryapi
 * \ingroup module_mdtypes
 */
#ifndef GMX_MDTYPES_IFORCEPROVIDER_H
#define GMX_MDTYPES_IFORCEPROVIDER_H

#include <cstdint>

#include <memory>
#include <string>

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

struct gmx_domdec_t;
struct gmx_enerdata_t;
struct gmx_wallcycle;
struct t_forcerec;
struct interaction_const_t;
class GpuEventSynchronizer;

namespace gmx
{

template<typename T>
class ArrayRef;
class ForceWithVirial;
class MpiComm;


/*! \libinternal \brief
 * Helper struct that bundles data for passing it over to the force providers
 *
 * This is a short-lived container that bundles up all necessary input data for the
 * force providers. Its only purpose is to allow calling forceProviders->calculateForces()
 * with just two arguments, one being the container for the input data,
 * the other the container for the output data.
 *
 * Both ForceProviderInput as well as ForceProviderOutput only package existing
 * data structs together for handing it over to calculateForces(). Apart from the
 * POD entries they own nothing.
 */
class ForceProviderInput
{
public:
    /*! \brief Constructor assembles all necessary force provider input data
     *
     * \param[in]  x        Atomic positions.
     * \param[in]  homenr   Number of atoms on the domain.
     * \param[in]  chargeA  Atomic charges for atoms on the domain.
     * \param[in]  massT    Atomic masses for atoms on the domain.
     * \param[in]  time     The current time in the simulation.
     * \param[in]  step     The current step in the simulation
     * \param[in]  box      The simulation box.
     * \param[in]  mpiComm  Communication object for my group.
     * \param[in]  dd       Domain decomposition object, pass nullptr when DD is not in use.
     */
    ForceProviderInput(ArrayRef<const RVec> x,
                       int                  homenr,
                       ArrayRef<const real> chargeA,
                       ArrayRef<const real> massT,
                       double               time,
                       int64_t              step,
                       const matrix         box,
                       const MpiComm&       mpiComm,
                       const gmx_domdec_t*  dd,
                       DeviceBuffer<RVec>  deviceCoordinates = {},
                       GpuEventSynchronizer* deviceCoordinatesReady = nullptr) :
        x_(x),
        homenr_(homenr),
        chargeA_(chargeA),
        massT_(massT),
        t_(time),
        step_(step),
        mpiComm_(mpiComm),
        dd_(dd),
        deviceCoordinates_(deviceCoordinates),
        deviceCoordinatesReady_(deviceCoordinatesReady)
    {
        copy_mat(box, box_);
    }

    ArrayRef<const RVec> x_; //!< The atomic positions
    int                  homenr_;
    ArrayRef<const real> chargeA_;
    ArrayRef<const real> massT_;
    double               t_;    //!< The current time in the simulation
    int64_t              step_; //!< The current step in the simulation
    matrix               box_ = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }; //!< The simulation box
    const MpiComm&       mpiComm_; //!< Communication object for my group
    const gmx_domdec_t*  dd_;      //!< Domain decomposition object, deprecated
    DeviceBuffer<RVec>   deviceCoordinates_; //!< Optional GROMACS device coordinates
    GpuEventSynchronizer* deviceCoordinatesReady_; //!< Readiness event for device coordinates

    //! Electrostatics interaction constants (ewaldCoeff, ewaldShift, epsfac, cutoffs),
    //! or nullptr when not available.
    const interaction_const_t* ic_ = nullptr;

    //! Per-atom reciprocal-space Coulomb potential computed by PME during this
    //! step (device memory, size homenr_). Empty when not available.
    DeviceBuffer<float> coulombPotential_;

    //! Event signaling that \p coulombPotential_ is ready on the device.
    GpuEventSynchronizer* coulombPotentialReady_ = nullptr;

    //! Whether the potential energy is needed this step (energy output,
    //! energy-based algorithms).
    bool computeEnergy = false;

    //! Whether the virial is needed this step.
    bool computeVirial = false;
};

/*! \brief Take pointer, check if valid, return reference
 */
template<class T>
T& makeRefFromPointer(T* ptr)
{
    GMX_ASSERT(ptr != nullptr, "got null pointer");
    return *ptr;
}

/*! \libinternal \brief
 * Helper struct bundling the output data of a force provider
 *
 * Same as for the ForceProviderInput class, but these variables can be written as well.
 */
class ForceProviderOutput
{
public:
    /*! \brief Constructor assembles all necessary force provider output data
     *
     * \param[in,out]  forceWithVirial  Container for force and virial
     * \param[in,out]  enerd            Structure containing energy data
     */
    ForceProviderOutput(ForceWithVirial* forceWithVirial, gmx_enerdata_t* enerd) :
        forceWithVirial_(makeRefFromPointer(forceWithVirial)), enerd_(makeRefFromPointer(enerd))
    {
    }

    ForceWithVirial& forceWithVirial_; //!< Container for force and virial
    gmx_enerdata_t&  enerd_;           //!< Structure containing energy data
};


/*! \libinternal \brief
 * Interface for a component that provides forces during MD.
 *
 * Modules implementing IMDModule generally implement this internally, and use
 * IMDModule::initForceProviders() to register their implementation in
 * ForceProviders.
 *
 * The interface most likely requires additional generalization for use in
 * other modules than the current electric field implementation.
 *
 * The forces that are produced by force providers are not taken into account
 * in the calculation of the virial. When applicable, the provider should
 * compute its own virial contribution.
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */
class IForceProvider
{
public:
    /*! \brief
     * Computes forces.
     *
     * \param[in]    forceProviderInput    struct that collects input data for the force providers
     * \param[in,out] forceProviderOutput   struct that collects output data of the force providers
     */
    virtual void calculateForces(const ForceProviderInput& forceProviderInput,
                                 ForceProviderOutput*      forceProviderOutput) = 0;

    //! Whether this provider can produce forces directly in a device RVec buffer.
    virtual bool supportsDeviceForces() const { return false; }

    //! Device force buffer produced by the most recent calculateForces call.
    virtual DeviceBuffer<RVec> deviceForceBuffer() const { return {}; }

    /*! \brief
     * Whether the device force buffer currently covers the local atoms.
     *
     * Device force providers may fall back to the host force path on a
     * per-step basis; in that case the device buffer must not be consumed.
     */
    virtual bool deviceForcesValidForCurrentStep() const { return false; }

    /*! \brief
     * Whether this provider updates the atomic charges every step.
     *
     * When true, updateCharges() is called once per step after the
     * coordinates have been finalized and before the non-bonded/PME force
     * computation.
     */
    virtual bool supportsDynamicCharges() const { return false; }

    /*! \brief
     * Whether this provider consumes the per-atom Coulomb potential computed
     * by GROMACS during the non-bonded/PME force evaluation.
     */
    virtual bool needsCoulombPotential() const { return false; }

    /*! \brief
     * Updates the atomic charges before the non-bonded/PME force computation.
     *
     * Providers may overwrite the values in \p chargeA; the MD loop re-uploads
     * the charges to the non-bonded and PME GPU data structures when this
     * method returns true.
     *
     * \param[in]     x                 Atomic positions (host).
     * \param[in]     deviceCoordinates GROMACS device coordinates.
     * \param[in]     deviceCoordinatesReady Event signaling device coordinate readiness.
     * \param[in,out] chargeA           Atomic charges to update (size homenr).
     * \param[in]     ic                Electrostatics interaction constants.
     * \param[in]     box               The simulation box.
     * \param[in]     homenr            Number of atoms on the domain.
     * \returns whether the charges changed and need re-uploading.
     */
    virtual bool updateCharges(ArrayRef<const RVec> /*x*/,
                               DeviceBuffer<RVec> /*deviceCoordinates*/,
                               GpuEventSynchronizer* /*deviceCoordinatesReady*/,
                               ArrayRef<real> /*chargeA*/,
                               const interaction_const_t& /*ic*/,
                               const matrix /*box*/,
                               int /*homenr*/)
    {
        return false;
    }

protected:
    ~IForceProvider() {}
};

/*! \libinternal \brief
 * Evaluates forces from a collection of gmx::IForceProvider.
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */
class ForceProviders
{
public:
    /*! \brief
     * Constructor.
     *
     * \param[in] wallCycle  Pointer to a wallcycle counter struct, can be nullptr
     */
    ForceProviders(gmx_wallcycle* wallCycle = nullptr);
    ~ForceProviders();

    /*! \brief
     * Adds a provider.
     *
     * \param[in] provider          The force provider callback function
     * \param[in] cycleCounterName  A non-empty string will add a cycle counter with the given name
     *                              that registers the time spent in the force provider function
     */
    void addForceProvider(gmx::IForceProvider* provider, const std::string& cycleCounterName = "");

    //! Whether there are modules added.
    bool hasForceProvider() const;

    //! Whether all registered providers can use the device-force path.
    bool allForceProvidersSupportDeviceForces() const;

    //! Device force buffer produced by the device-capable provider.
    DeviceBuffer<RVec> deviceForceBuffer() const;

    //! Whether the device force buffer covers the local atoms for this step.
    bool deviceForcesValidForCurrentStep() const;

    //! Whether any registered provider updates the atomic charges every step.
    bool haveDynamicCharges() const;

    //! Whether any registered provider needs the per-atom Coulomb potential.
    bool needCoulombPotential() const;

    /*! \brief
     * Updates the atomic charges of the providers that support dynamic charges.
     *
     * \returns whether any provider updated the charges.
     */
    bool updateCharges(ArrayRef<const RVec>       x,
                       DeviceBuffer<RVec>         deviceCoordinates,
                       GpuEventSynchronizer*      deviceCoordinatesReady,
                       ArrayRef<real>             chargeA,
                       const interaction_const_t& ic,
                       const matrix               box,
                       int                        homenr) const;

    //! Computes forces.
    void calculateForces(const gmx::ForceProviderInput& forceProviderInput,
                         gmx::ForceProviderOutput*      forceProviderOutput) const;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
