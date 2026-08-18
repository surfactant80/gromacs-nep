/*
 * GROMACS qNEP adapter for GROMACS-owned electrostatics.
 *
 * This file is part of a local GROMACS/GPUMD integration. The NEP evaluator
 * kernels below are derived from GPUMD (src/external/gpumd) and remain under
 * the GPUMD license. They are duplicated here, outside the vendored tree, so
 * that the electrostatics part can be removed and replaced by the GROMACS
 * non-bonded/PME computation while the vendored GPUMD tree stays unmodified.
 *
 * The qNEP descriptor, charge and force kernels are copies of the GPUMD
 * NEP_Charge kernels with the following changes:
 *   - the PPPM/Ewald calls are removed;
 *   - the charge-response force uses an externally supplied per-atom Coulomb
 *     potential (D_real) computed by GROMACS;
 *   - a charge-mode-2 variant of the real-space kernel subtracts the GPUMD
 *     style real-space electrostatics from energy/forces/virial/D_real.
 */

// Pre-include all standard-library headers that the GPUMD headers may pull in,
// so that the temporary "private -> public" macro below can never reach them.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// The vendored NEP_Charge keeps its parameters and buffers private. This
// adapter reuses them (parser, parameter layout, neighbor lists, descriptor
// data) without modifying the vendored source, by making the members
// accessible in this translation unit only.
#define private public
#define protected public
#include "force/nep_charge.cuh"
#undef protected
#undef private

#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"

#include "nepmodel_qnep_gromacs.h"

namespace
{

constexpr int c_blockSize = 64;

// Coefficient of the point-charge self term in the charge-response
// potential: -SELF_FACTOR * beta/sqrt(pi) * q. The Ewald self energy is
// -beta/sqrt(pi) * sum(q^2), whose derivative is -2 * beta/sqrt(pi) * q.
constexpr float SELF_FACTOR = 2.0f;

// Copied from GPUMD nep_charge.cu: filter the global neighbor list into the
// radial and angular neighbor lists.
static __global__ void find_neighbor_list_large_box_gmx(
        NEP_Charge::ParaMB paramb,
        const int          N,
        const int          N1,
        const int          N2,
        const Box          box,
        const int*         g_type,
        const double* __restrict__ g_x,
        const double* __restrict__ g_y,
        const double* __restrict__ g_z,
        const int* __restrict__ g_NN_global,
        const int* __restrict__ g_NL_global,
        int* g_NN_radial,
        int* g_NL_radial,
        int* g_NN_angular,
        int* g_NL_angular)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N2)
    {
        return;
    }

    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int    count_radial  = 0;
    int    count_angular = 0;

    for (int i1 = 0; i1 < g_NN_global[n1]; ++i1)
    {
        int   n2   = g_NL_global[n1 + N * i1];
        float x12  = g_x[n2] - x1;
        float y12  = g_y[n2] - y1;
        float z12  = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
        float rc_radial  = paramb.rc_radial;
        float rc_angular = paramb.rc_angular;
        if (d12_square >= rc_radial * rc_radial)
        {
            continue;
        }
        g_NL_radial[count_radial++ * N + n1] = n2;
        if (d12_square < rc_angular * rc_angular)
        {
            g_NL_angular[count_angular++ * N + n1] = n2;
        }
    }

    g_NN_radial[n1]  = count_radial;
    g_NN_angular[n1] = count_angular;
}

// Copied from GPUMD nep_charge.cu: descriptors, short-range energy, charges
// and charge derivatives. No electrostatics is computed here.
static __global__ void find_descriptor_gmx(
        NEP_Charge::ParaMB paramb,
        NEP_Charge::ANN    annmb,
        const int          N,
        const int          N1,
        const int          N2,
        const Box          box,
        const int*         g_NN,
        const int*         g_NL,
        const int*         g_NN_angular,
        const int*         g_NL_angular,
        const int* __restrict__ g_type,
        const double* __restrict__ g_x,
        const double* __restrict__ g_y,
        const double* __restrict__ g_z,
        double* g_pe,
        float*  g_Fp,
        float*  g_charge,
        float*  g_charge_derivative,
        double* g_virial,
        float*  g_sum_fxyz)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 < N2)
    {
        int    t1 = g_type[n1];
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        float  q[MAX_DIM] = { 0.0f };

        // get radial descriptors
        for (int i1 = 0; i1 < g_NN[n1]; ++i1)
        {
            int   n2  = g_NL[n1 + N * i1];
            float x12 = g_x[n2] - x1;
            float y12 = g_y[n2] - y1;
            float z12 = g_z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            float d12  = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            float fc12;
            int   t2 = g_type[n2];
            float rc = paramb.rc_radial;
            float rcinv = 1.0f / rc;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];

            find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
            for (int n = 0; n <= paramb.n_max_radial; ++n)
            {
                float gn12 = 0.0f;
                for (int k = 0; k <= paramb.basis_size_radial; ++k)
                {
                    int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
                    c_index += t1 * paramb.num_types + t2;
                    gn12 += fn12[k] * annmb.c[c_index];
                }
                q[n] += gn12;
            }
        }

        // get angular descriptors
        for (int n = 0; n <= paramb.n_max_angular; ++n)
        {
            float s[NUM_OF_ABC] = { 0.0f };
            for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1)
            {
                int   n2  = g_NL_angular[n1 + N * i1];
                float x12 = g_x[n2] - x1;
                float y12 = g_y[n2] - y1;
                float z12 = g_z[n2] - z1;
                apply_mic(box, x12, y12, z12);
                float d12  = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
                float fc12;
                int   t2 = g_type[n2];
                float rc = paramb.rc_angular;
                float rcinv = 1.0f / rc;
                find_fc(rc, rcinv, d12, fc12);
                float fn12[MAX_NUM_N];
                find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
                float gn12 = 0.0f;
                for (int k = 0; k <= paramb.basis_size_angular; ++k)
                {
                    int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
                    c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
                    gn12 += fn12[k] * annmb.c[c_index];
                }
                accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
            }
            find_q(paramb.L_max,
                   paramb.has_q_222,
                   paramb.has_q_1111,
                   paramb.has_q_112,
                   paramb.has_q_123,
                   paramb.has_q_233,
                   paramb.has_q_134,
                   paramb.n_max_angular + 1,
                   n,
                   s,
                   q + (paramb.n_max_radial + 1));
            for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc)
            {
                g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] =
                        s[abc];
            }
        }

        // normalize descriptor
        for (int d = 0; d < annmb.dim; ++d)
        {
            q[d] = q[d] * annmb.q_scaler[d];
        }

        float F                 = 0.0f;
        float Fp[MAX_DIM]       = { 0.0f };
        float charge            = 0.0f;
        float charge_derivative[MAX_DIM] = { 0.0f };

        apply_ann_one_layer_charge(annmb.dim,
                                   annmb.num_neurons1,
                                   annmb.w0[t1],
                                   annmb.b0[t1],
                                   annmb.w1[t1],
                                   annmb.b1,
                                   q,
                                   F,
                                   Fp,
                                   charge,
                                   charge_derivative);

        g_pe[n1]     += F;
        g_charge[n1] = charge;

        for (int d = 0; d < annmb.dim; ++d)
        {
            g_Fp[d * N + n1]                = Fp[d] * annmb.q_scaler[d];
            g_charge_derivative[d * N + n1] = charge_derivative[d] * annmb.q_scaler[d];
        }
    }
}

// Copied from GPUMD nep_charge.cu: enforce charge neutrality by subtracting
// the mean charge.
static __global__ void zero_total_charge_gmx(const int N, float* g_charge)
{
    int tid = threadIdx.x;
    int number_of_batches = (N - 1) / 1024 + 1;
    __shared__ float s_charge[1024];
    float charge = 0.0f;
    for (int batch = 0; batch < number_of_batches; ++batch)
    {
        int n = tid + batch * 1024;
        if (n < N)
        {
            charge += g_charge[n];
        }
    }
    s_charge[tid] = charge;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_charge[tid] += s_charge[tid + offset];
        }
        __syncthreads();
    }

    for (int batch = 0; batch < number_of_batches; ++batch)
    {
        int n = tid + batch * 1024;
        if (n < N)
        {
            g_charge[n] -= s_charge[0] / N;
        }
    }
}

// Fills the D_real (charge-response potential) buffer with the GROMACS
// per-atom Coulomb potential (reciprocal + self + short-range pair part),
// converted from kJ/(mol e) to eV/e. Units: GROMACS beta in 1/nm, ewaldShift
// in 1/nm, epsfac in kJ nm/(mol e^2); distances in Angstrom. The reciprocal
// part is the B-spline-interpolated potential from the PME gather. The pair
// part uses the NEP global neighbor list, which for this adapter is built at
// rc = rcoulomb, so every pair is within the GROMACS Coulomb cutoff and there
// are no exclusions (the qNEP mode clears all exclusions).
static __global__ void fill_d_real_gmx(const int   N,
                                       const float* __restrict__ g_pme_potential,
                                       const int* __restrict__ g_NN,
                                       const int* __restrict__ g_NL,
                                       const float* __restrict__ g_charge,
                                       const double* __restrict__ g_x,
                                       const double* __restrict__ g_y,
                                       const double* __restrict__ g_z,
                                       const Box    box,
                                       const float  betaPerAngstrom,
                                       const float  ewaldShiftPerAngstrom,
                                       const float  epsfac,
                                       const float  rcoulombPerAngstrom,
                                       float* __restrict__ g_d_real)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        const double x1 = g_x[n1];
        const double y1 = g_y[n1];
        const double z1 = g_z[n1];
        const float  q1 = g_charge[n1];

        // epsfac is passed in kJ nm/(mol e^2) but the distances here are in
        // Angstrom, so the effective constant is 10 * epsfac.
        const float epsfacAngstrom = 10.0f * epsfac;

        // Reciprocal + self term:
        //   phi_rec from the PME gather;
        //   self term: -2 * epsfac * beta / sqrt(pi) * q1 (kJ/(mol e)).
        float potential = g_pme_potential[n1]
                          - SELF_FACTOR * epsfacAngstrom * betaPerAngstrom / sqrt(float(PI)) * q1;

        const float rcoulombSquared = rcoulombPerAngstrom * rcoulombPerAngstrom;
        for (int i1 = 0; i1 < g_NN[n1]; ++i1)
        {
            const int n2 = g_NL[n1 + N * i1];
            float     x12 = g_x[n2] - x1;
            float     y12 = g_y[n2] - y1;
            float     z12 = g_z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            const float d12Squared = x12 * x12 + y12 * y12 + z12 * z12;
            if (d12Squared >= rcoulombSquared)
            {
                continue;
            }
            const float d12 = sqrt(d12Squared);
            potential += epsfacAngstrom * g_charge[n2]
                         * (erfc(betaPerAngstrom * d12) / d12 - ewaldShiftPerAngstrom);
        }

        g_d_real[n1] = potential / 96.48533212331002f;
    }
}

// Copied from GPUMD nep_charge.cu: chain-rule correction. zero_total_charge
// shifted the charges by -mean(q), so D_real must be shifted by -mean(D_real)
// for consistent forces. Uses a double accumulator for numerical precision.
static __global__ void zero_mean_D_real_gmx(const int N, float* g_d_real)
{
    int tid = threadIdx.x;
    int number_of_batches = (N - 1) / 1024 + 1;
    __shared__ double s_sum[1024];
    double sum = 0.0;
    for (int batch = 0; batch < number_of_batches; ++batch)
    {
        int n = tid + batch * 1024;
        if (n < N)
        {
            sum += static_cast<double>(g_d_real[n]);
        }
    }
    s_sum[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }

    float mean_D = static_cast<float>(s_sum[0] / N);
    for (int batch = 0; batch < number_of_batches; ++batch)
    {
        int n = tid + batch * 1024;
        if (n < N)
        {
            g_d_real[n] -= mean_D;
        }
    }
}

// Copied from GPUMD nep_charge.cu: radial force kernel with the charge-
// response chain-rule term (charge_derivative * D_real).
static __global__ void find_force_radial_gmx(
        NEP_Charge::ParaMB paramb,
        NEP_Charge::ANN    annmb,
        const int          N,
        const int          N1,
        const int          N2,
        const Box          box,
        const int*         g_NN,
        const int*         g_NL,
        const int* __restrict__ g_type,
        const double* __restrict__ g_x,
        const double* __restrict__ g_y,
        const double* __restrict__ g_z,
        const float* __restrict__ g_Fp,
        const float* g_charge_derivative,
        const float* g_D_real,
        double* g_fx,
        double* g_fy,
        double* g_fz,
        double* g_virial)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 < N2)
    {
        int   t1 = g_type[n1];
        float s_fx  = 0.0f;
        float s_fy  = 0.0f;
        float s_fz  = 0.0f;
        float s_sxx = 0.0f;
        float s_sxy = 0.0f;
        float s_sxz = 0.0f;
        float s_syx = 0.0f;
        float s_syy = 0.0f;
        float s_syz = 0.0f;
        float s_szx = 0.0f;
        float s_szy = 0.0f;
        float s_szz = 0.0f;
        double x1   = g_x[n1];
        double y1   = g_y[n1];
        double z1   = g_z[n1];
        for (int i1 = 0; i1 < g_NN[n1]; ++i1)
        {
            int   n2 = g_NL[n1 + N * i1];
            int   t2 = g_type[n2];
            float x12 = g_x[n2] - x1;
            float y12 = g_y[n2] - y1;
            float z12 = g_z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            float r12[3] = { x12, y12, z12 };
            float d12    = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            float d12inv = 1.0f / d12;
            float f12[3] = { 0.0f };
            float f21[3] = { 0.0f };
            float fc12, fcp12;
            float rc    = paramb.rc_radial;
            float rcinv = 1.0f / rc;
            find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
            float fn12[MAX_NUM_N];
            float fnp12[MAX_NUM_N];
            find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
            for (int n = 0; n <= paramb.n_max_radial; ++n)
            {
                float gnp12 = 0.0f;
                float gnp21 = 0.0f;
                for (int k = 0; k <= paramb.basis_size_radial; ++k)
                {
                    int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
                    gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
                    gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
                }
                float tmp12 = g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1];
                float tmp21 = g_Fp[n2 + n * N] + g_charge_derivative[n2 + n * N] * g_D_real[n2];
                tmp12 *= gnp12 * d12inv;
                tmp21 *= gnp21 * d12inv;
                for (int d = 0; d < 3; ++d)
                {
                    f12[d] += tmp12 * r12[d];
                    f21[d] -= tmp21 * r12[d];
                }
            }
            s_fx += f12[0] - f21[0];
            s_fy += f12[1] - f21[1];
            s_fz += f12[2] - f21[2];
            s_sxx += r12[0] * f21[0];
            s_syy += r12[1] * f21[1];
            s_szz += r12[2] * f21[2];
            s_sxy += r12[0] * f21[1];
            s_sxz += r12[0] * f21[2];
            s_syx += r12[1] * f21[0];
            s_syz += r12[1] * f21[2];
            s_szx += r12[2] * f21[0];
            s_szy += r12[2] * f21[1];
        }
        g_fx[n1] += s_fx;
        g_fy[n1] += s_fy;
        g_fz[n1] += s_fz;
        g_virial[n1 + 0 * N] += s_sxx;
        g_virial[n1 + 1 * N] += s_syy;
        g_virial[n1 + 2 * N] += s_szz;
        g_virial[n1 + 3 * N] += s_sxy;
        g_virial[n1 + 4 * N] += s_sxz;
        g_virial[n1 + 5 * N] += s_syz;
        g_virial[n1 + 6 * N] += s_syx;
        g_virial[n1 + 7 * N] += s_szx;
        g_virial[n1 + 8 * N] += s_szy;
    }
}

// Copied from GPUMD nep_charge.cu: angular partial force kernel with the
// charge-response chain-rule term.
static __global__ void find_partial_force_angular_gmx(
        NEP_Charge::ParaMB paramb,
        NEP_Charge::ANN    annmb,
        const int          N,
        const int          N1,
        const int          N2,
        const Box          box,
        const int*         g_NN_angular,
        const int*         g_NL_angular,
        const int* __restrict__ g_type,
        const double* __restrict__ g_x,
        const double* __restrict__ g_y,
        const double* __restrict__ g_z,
        const float* __restrict__ g_Fp,
        const float* g_charge_derivative,
        const float* g_D_real,
        const float* __restrict__ g_sum_fxyz,
        float* g_f12x,
        float* g_f12y,
        float* g_f12z)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 < N2)
    {
        float Fp[MAX_DIM_ANGULAR]          = { 0.0f };
        float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
        for (int d = 0; d < paramb.dim_angular; ++d)
        {
            float tmp = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1]
                        + g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1]
                                  * g_D_real[n1];
            Fp[d] = tmp;
        }
        for (int n = 0; n < paramb.n_max_angular + 1; ++n)
        {
            for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc)
            {
                sum_fxyz[n * NUM_OF_ABC + abc] =
                        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N
                                   + n1];
            }
        }

        int    t1 = g_type[n1];
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1)
        {
            int   index = i1 * N + n1;
            int   n2    = g_NL_angular[n1 + N * i1];
            float x12   = g_x[n2] - x1;
            float y12   = g_y[n2] - y1;
            float z12   = g_z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            float r12[3] = { x12, y12, z12 };
            float d12    = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            float f12[3] = { 0.0f };
            float fc12, fcp12;
            int   t2 = g_type[n2];
            float rc = paramb.rc_angular;
            float rcinv = 1.0f / rc;
            find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

            float fn12[MAX_NUM_N];
            float fnp12[MAX_NUM_N];
            find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
            for (int n = 0; n <= paramb.n_max_angular; ++n)
            {
                float gn12  = 0.0f;
                float gnp12 = 0.0f;
                for (int k = 0; k <= paramb.basis_size_angular; ++k)
                {
                    int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
                    c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
                    gn12 += fn12[k] * annmb.c[c_index];
                    gnp12 += fnp12[k] * annmb.c[c_index];
                }
                accumulate_f12(paramb.L_max,
                               paramb.has_q_222,
                               paramb.has_q_1111,
                               paramb.has_q_112,
                               paramb.has_q_123,
                               paramb.has_q_233,
                               paramb.has_q_134,
                               paramb.num_L,
                               n,
                               paramb.n_max_angular + 1,
                               d12,
                               r12,
                               gn12,
                               gnp12,
                               Fp,
                               sum_fxyz,
                               f12);
            }
            g_f12x[index] = f12[0];
            g_f12y[index] = f12[1];
            g_f12z[index] = f12[2];
        }
    }
}

// Copied from GPUMD nep_charge.cu (find_force_charge_real_space) with a sign
// parameter. For charge mode 2, the GPUMD-style real-space electrostatics
// (alpha = pi/rc_radial) is subtracted from the GROMACS total Coulomb energy,
// forces, virial and from D_real (the chain-rule potential).
template<bool subtract>
static __global__ void find_force_charge_real_space_gmx(
        const int                     N,
        const NEP_Charge::Charge_Para charge_para,
        const int                     N1,
        const int                     N2,
        const Box                     box,
        const int*                    g_NN,
        const int*                    g_NL,
        const float*                  g_charge,
        const double* __restrict__ g_x,
        const double* __restrict__ g_y,
        const double* __restrict__ g_z,
        double* g_fx,
        double* g_fy,
        double* g_fz,
        double* g_virial,
        double* g_pe,
        float*  g_D_real)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 < N2)
    {
        float s_fx  = 0.0f;
        float s_fy  = 0.0f;
        float s_fz  = 0.0f;
        float s_sxx = 0.0f;
        float s_sxy = 0.0f;
        float s_sxz = 0.0f;
        float s_syx = 0.0f;
        float s_syy = 0.0f;
        float s_syz = 0.0f;
        float s_szx = 0.0f;
        float s_szy = 0.0f;
        float s_szz = 0.0f;
        double x1   = g_x[n1];
        double y1   = g_y[n1];
        double z1   = g_z[n1];
        float  q1   = g_charge[n1];
        float s_pe  = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1; // self energy part
        float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi;            // self energy part

        for (int i1 = 0; i1 < g_NN[n1]; ++i1)
        {
            int   n2 = g_NL[n1 + N * i1];
            float q2 = g_charge[n2];
            float qq = q1 * q2;
            float x12 = g_x[n2] - x1;
            float y12 = g_y[n2] - y1;
            float z12 = g_z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            float r12[3] = { x12, y12, z12 };
            float d12    = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            float d12inv = 1.0f / d12;

            float erfc_r = erfc(charge_para.alpha * d12) * d12inv;
            D_real += q2 * erfc_r;
            s_pe += 0.5f * qq * erfc_r;
            float f2 = erfc_r
                       + charge_para.two_alpha_over_sqrt_pi
                                 * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
            f2 *= -0.5f * K_C_SP * qq * d12inv * d12inv;
            float f12[3] = { r12[0] * f2, r12[1] * f2, r12[2] * f2 };
            float f21[3] = { -r12[0] * f2, -r12[1] * f2, -r12[2] * f2 };

            s_fx += f12[0] - f21[0];
            s_fy += f12[1] - f21[1];
            s_fz += f12[2] - f21[2];
            s_sxx -= r12[0] * f12[0];
            s_sxy -= r12[0] * f12[1];
            s_sxz -= r12[0] * f12[2];
            s_syx -= r12[1] * f12[0];
            s_syy -= r12[1] * f12[1];
            s_syz -= r12[1] * f12[2];
            s_szx -= r12[2] * f12[0];
            s_szy -= r12[2] * f12[1];
            s_szz -= r12[2] * f12[2];
        }
        if constexpr (subtract)
        {
            g_fx[n1] -= s_fx;
            g_fy[n1] -= s_fy;
            g_fz[n1] -= s_fz;
            g_virial[n1 + 0 * N] -= s_sxx;
            g_virial[n1 + 1 * N] -= s_syy;
            g_virial[n1 + 2 * N] -= s_szz;
            g_virial[n1 + 3 * N] -= s_sxy;
            g_virial[n1 + 4 * N] -= s_sxz;
            g_virial[n1 + 5 * N] -= s_syz;
            g_virial[n1 + 6 * N] -= s_syx;
            g_virial[n1 + 7 * N] -= s_szx;
            g_virial[n1 + 8 * N] -= s_szy;
            g_D_real[n1] -= K_C_SP * D_real;
            g_pe[n1] -= K_C_SP * s_pe;
        }
        else
        {
            g_fx[n1] += s_fx;
            g_fy[n1] += s_fy;
            g_fz[n1] += s_fz;
            g_virial[n1 + 0 * N] += s_sxx;
            g_virial[n1 + 1 * N] += s_syy;
            g_virial[n1 + 2 * N] += s_szz;
            g_virial[n1 + 3 * N] += s_sxy;
            g_virial[n1 + 4 * N] += s_sxz;
            g_virial[n1 + 5 * N] += s_syz;
            g_virial[n1 + 6 * N] += s_syx;
            g_virial[n1 + 7 * N] += s_szx;
            g_virial[n1 + 8 * N] += s_szy;
            g_D_real[n1] += K_C_SP * D_real;
            g_pe[n1] += K_C_SP * s_pe;
        }
    }
}

} // namespace

namespace gmx
{

class NepQnepGromacs::Impl
{
public:
    Impl(NEP_Charge* model, int chargeMode, float rcCoulombA, int numAtoms) :
        model_(model),
        chargeMode_(chargeMode),
        rcCoulombA_(rcCoulombA)
    {
        // The model's global neighbor list is reused for both the NEP
        // descriptor filtering and the short-range potential sum, so it is
        // searched at the (larger) Coulomb cutoff. Re-initialize it with
        // enough capacity: GPUMD scales the capacity by (rc + skin)^3/rc^3
        // with skin = 1 A, and the sort kernel uses one block of MN threads,
        // so MN must stay below 1024.
        const int numNeighbors = 600;
        model_->neighbor.initialize(rcCoulombA, numAtoms, numNeighbors);
    }

    NEP_Charge* model_;
    int         chargeMode_;
    float       rcCoulombA_;
    bool        useCachedSearch_ = false;
    bool        searchModeSet_   = false;
};

NepQnepGromacs::NepQnepGromacs(NEP_Charge* model, int chargeMode, float rcCoulombA, int numAtoms) :
    impl_(std::make_unique<Impl>(model, chargeMode, rcCoulombA, numAtoms))
{}

NepQnepGromacs::~NepQnepGromacs() = default;

void NepQnepGromacs::predictCharges(Box&                      box,
                                    const GPU_Vector<int>&    types,
                                    const GPU_Vector<double>& positions,
                                    float*                    hostCharges,
                                    GPU_Vector<double>&       potential,
                                    int                       numAtoms)
{
    NEP_Charge* model    = impl_->model_;
    const int   N        = numAtoms;
    const int   N1       = model->N1;
    const int   N2       = model->N2;
    const int   gridSize = (N2 - N1 - 1) / c_blockSize + 1;

    // Decide once whether the box is large enough for the cached search
    // (search radius = rcoulomb + 1 A skin, five cells required).
    if (!impl_->searchModeSet_)
    {
        const double volume = box.get_volume();
        const double thicknessX = volume / box.get_area(0);
        const double thicknessY = volume / box.get_area(1);
        const double thicknessZ = volume / box.get_area(2);
        const double minThickness = std::min({ thicknessX, thicknessY, thicknessZ });
        impl_->useCachedSearch_ = minThickness >= 2.5 * (impl_->rcCoulombA_ + 1.0);
        impl_->searchModeSet_ = true;
    }

    // 1) Global neighbor list. GPUMD's Neighbor searches at rc + skin with
    // skin = 1 A and rebuilds automatically when any atom has moved more than
    // 0.5 A, so the list is reused across steps. The cell-list search needs at
    // least five cells per periodic direction (box >= 2.5 * search radius);
    // for smaller boxes the list is rebuilt every step at the exact Coulomb
    // cutoff instead.
    if (impl_->useCachedSearch_)
    {
        model->neighbor.find_neighbor_global(impl_->rcCoulombA_, box, types, positions);
    }
    else
    {
        // Force the rebuild every step (bypass GPUMD's internal caching) and
        // compensate the internal +skin so the actual search radius equals the
        // Coulomb cutoff.
        model->neighbor.x0.resize(0);
        model->neighbor.find_neighbor_global(
                static_cast<double>(impl_->rcCoulombA_) - 1.0, box, types, positions);
    }

    // 2) Filter the radial and angular neighbor lists (always, so the
    // descriptor lists follow the current positions even with a reused global
    // list).
    find_neighbor_list_large_box_gmx<<<gridSize, c_blockSize>>>(
            model->paramb,
            N,
            N1,
            N2,
            box,
            types.data(),
            positions.data(),
            positions.data() + N,
            positions.data() + N * 2,
            model->neighbor.NN.data(),
            model->neighbor.NL.data(),
            model->nep_data.NN_radial.data(),
            model->nep_data.NL_radial.data(),
            model->nep_data.NN_angular.data(),
            model->nep_data.NL_angular.data());
    GPU_CHECK_KERNEL

    if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
    {
        CHECK(cudaDeviceSynchronize());
        std::vector<int> nn(N);
        model->neighbor.NN.copy_to_host(nn.data());
        std::vector<int> nnr(N);
        model->nep_data.NN_radial.copy_to_host(nnr.data());
        std::vector<int> nl(N * model->paramb.MN_radial);
        model->nep_data.NL_radial.copy_to_host(nl.data());
        fprintf(stderr, "[qnep-debug] global NN[0]=%d radial NN[0]=%d first: %d %d %d %d\n",
                nn[0], nnr[0], nl[0], nl[N], nl[2*N], nl[3*N]);
    }

    // 3) Descriptors: short-range energy, charges and charge derivatives.
    find_descriptor_gmx<<<gridSize, c_blockSize>>>(
            model->paramb,
            model->annmb,
            N,
            N1,
            N2,
            box,
            model->nep_data.NN_radial.data(),
            model->nep_data.NL_radial.data(),
            model->nep_data.NN_angular.data(),
            model->nep_data.NL_angular.data(),
            types.data(),
            positions.data(),
            positions.data() + N,
            positions.data() + N * 2,
            potential.data(),
            model->nep_data.Fp.data(),
            model->nep_data.charge.data(),
            model->nep_data.charge_derivative.data(),
            nullptr,
            model->nep_data.sum_fxyz.data());
    GPU_CHECK_KERNEL

    // 4) Enforce charge neutrality.
    zero_total_charge_gmx<<<1, 1024>>>(N, model->nep_data.charge.data());
    GPU_CHECK_KERNEL

    // 5) Copy the charges to the host.
    model->nep_data.charge.copy_to_host(hostCharges);
    CHECK(cudaDeviceSynchronize());
}

void NepQnepGromacs::computeForces(Box&                      box,
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
                                   int                       numAtoms)
{
    NEP_Charge* model    = impl_->model_;
    const int   N        = numAtoms;
    const int   N1       = model->N1;
    const int   N2       = model->N2;
    const int   gridSize = (N2 - N1 - 1) / c_blockSize + 1;

    float* dReal = model->nep_data.D_real.data();

    // 1) Construct the GROMACS per-atom Coulomb potential (reciprocal from
    // the PME gather, self term, and the short-range pair part over the
    // global neighbor list built at the Coulomb cutoff in phase 1).
    const float betaPerAngstrom       = betaNm / 10.0f;
    const float ewaldShiftPerAngstrom = ewaldShiftNm / 10.0f;
    fill_d_real_gmx<<<(N + c_blockSize - 1) / c_blockSize, c_blockSize>>>(
            N,
            d_pmePotential,
            model->neighbor.NN.data(),
            model->neighbor.NL.data(),
            model->nep_data.charge.data(),
            positions.data(),
            positions.data() + N,
            positions.data() + N * 2,
            box,
            betaPerAngstrom,
            ewaldShiftPerAngstrom,
            epsfac,
            rcCoulombNm * 10.0f,
            dReal);
    GPU_CHECK_KERNEL

    // 2) Chain-rule correction: the charges were shifted by -mean(q) for
    // neutrality, so the charge-response potential must be shifted by
    // -mean(D_real) to keep the forces consistent with the energy.
    zero_mean_D_real_gmx<<<1, 1024>>>(N, dReal);
    GPU_CHECK_KERNEL

    // 3) Charge mode 2: subtract the GPUMD-style real-space electrostatics
    // from D_real, the energy and the forces/virial.
    if (impl_->chargeMode_ == 2)
    {
        find_force_charge_real_space_gmx<true><<<gridSize, c_blockSize>>>(
                N,
                model->charge_para,
                N1,
                N2,
                box,
                model->nep_data.NN_radial.data(),
                model->nep_data.NL_radial.data(),
                model->nep_data.charge.data(),
                positions.data(),
                positions.data() + N,
                positions.data() + N * 2,
                forces.data(),
                forces.data() + N,
                forces.data() + N * 2,
                virial.data(),
                potential.data(),
                dReal);
        GPU_CHECK_KERNEL
    }

    // 4) Radial forces with the chain-rule term.
    find_force_radial_gmx<<<gridSize, c_blockSize>>>(
            model->paramb,
            model->annmb,
            N,
            N1,
            N2,
            box,
            model->nep_data.NN_radial.data(),
            model->nep_data.NL_radial.data(),
            types.data(),
            positions.data(),
            positions.data() + N,
            positions.data() + N * 2,
            model->nep_data.Fp.data(),
            model->nep_data.charge_derivative.data(),
            dReal,
            forces.data(),
            forces.data() + N,
            forces.data() + N * 2,
            virial.data());
    GPU_CHECK_KERNEL

    // 5) Angular partial forces with the chain-rule term.
    find_partial_force_angular_gmx<<<gridSize, c_blockSize>>>(
            model->paramb,
            model->annmb,
            N,
            N1,
            N2,
            box,
            model->nep_data.NN_angular.data(),
            model->nep_data.NL_angular.data(),
            types.data(),
            positions.data(),
            positions.data() + N,
            positions.data() + N * 2,
            model->nep_data.Fp.data(),
            model->nep_data.charge_derivative.data(),
            dReal,
            model->nep_data.sum_fxyz.data(),
            model->nep_data.f12x.data(),
            model->nep_data.f12y.data(),
            model->nep_data.f12z.data());
    GPU_CHECK_KERNEL

    // 6) Reduce the angular many-body forces and virial.
    model->find_properties_many_body(box,
                                     model->nep_data.NN_angular.data(),
                                     model->nep_data.NL_angular.data(),
                                     model->nep_data.f12x.data(),
                                     model->nep_data.f12y.data(),
                                     model->nep_data.f12z.data(),
                                     false,
                                     positions,
                                     forces,
                                     virial);

    if (std::getenv("GMX_QNEP_DEBUG") != nullptr)
    {
        CHECK(cudaDeviceSynchronize());
        const int dim = model->annmb.dim;
        std::vector<float> cd(dim * N);
        std::vector<float> fp(dim * N);
        std::vector<float> ch(N);
        std::vector<float> dr(N);
        model->nep_data.charge_derivative.copy_to_host(cd.data());
        model->nep_data.Fp.copy_to_host(fp.data());
        model->nep_data.charge.copy_to_host(ch.data());
        model->nep_data.D_real.copy_to_host(dr.data());
        double cdSum = 0, fpSum = 0, chSum = 0, drSum = 0;
        for (int i = 0; i < dim * N; ++i)
        {
            cdSum += std::abs(cd[i]);
            fpSum += std::abs(fp[i]);
        }
        for (int i = 0; i < N; ++i)
        {
            chSum += std::abs(ch[i]);
            drSum += std::abs(dr[i]);
        }
        double pmeSum = 0;
        if (d_pmePotential != nullptr)
        {
            std::vector<float> pme(N);
            CHECK(cudaMemcpy(pme.data(),
                             d_pmePotential,
                             N * sizeof(float),
                             cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; ++i)
            {
                pmeSum += std::abs(pme[i]);
            }
        }
        double chSgn = 0, drSgn = 0, pmeSgn = 0;
        for (int i = 0; i < N; ++i)
        {
            chSgn += ch[i];
            drSgn += dr[i];
        }
        if (d_pmePotential != nullptr)
        {
            std::vector<float> pme2(N);
            CHECK(cudaMemcpy(pme2.data(),
                             d_pmePotential,
                             N * sizeof(float),
                             cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; ++i)
            {
                pmeSgn += pme2[i];
            }
        }
        fprintf(stderr,
                "[qnep-debug-adapter] dim=%d mean|dq/dqdes|=%g mean|Fp|=%g "
                "mean|q|=%g sumD=%g sumPme=%g mean|D_real|=%g mean|pme_pot|=%g (kJ/mol/e)\n",
                dim,
                cdSum / (dim * N),
                fpSum / (dim * N),
                chSum / N,
                drSgn / N,
                pmeSgn / N,
                drSum / N,
                pmeSum / N);
    }

    CHECK(cudaDeviceSynchronize());
}

} // namespace gmx
