# GROMACS–GPUMD NEP/qNEP 集成版（中文声明）

## 简介

本软件在 **GROMACS 2026.3** 中集成了 **GPUMD** 的 NEP/qNEP 机器学习势：GPUMD 负责 NEP/qNEP 的能量、力与 virial 计算，GROMACS 保留 MD 积分、恒温恒压、LINCS/SETTLE、经典键角二面角、LJ、库仑与 GPU PME、轨迹与能量输出。完整 GPUMD 源码以 vendored 形式内嵌于 `src/external/gpumd`，无需修改即可整体替换升级。

## 主要功能

- **普通 NEP**：支持 `nep4`、`nep4_zbl`、`nep5`、`nep5_zbl`；全体系 GPU 直通（原生速度 ~98%）与静态 NEP–MM mechanical embedding；
- **qNEP**（`nep4_charge1/2` 及其 `_zbl`）两种模式：
  - 原生模式：使用 GPUMD 自带的动态电荷、电荷中性化与 PPPM/Ewald；
  - **静电完全交由 GROMACS**（`nnpot-gromacs-charges = yes`）：qNEP 预测动态电荷 → 更新 GROMACS 非键/PME 电荷缓冲 → GROMACS 计算库仑能与力 → GPU PME 输出每原子倒空间静电势 → 链式力 `(dE/dq)(dq/dR)`；
- **动态 NEP–MM**：A/B 双组自适应 Δ-NEP（cross-only / a-nep-cross），自动选择活跃分子、平滑切换；
- **固定电荷长程方案**（`nnpot-use-gromacs-electrostatics = yes`）与**能量零点校准**（`nnpot-energy-offsets`）；
- **免显卡 grompp**：模型文件 CPU 端校验、懒加载，无 GPU 也能生成 tpr；
- **一键构建**：`./build-nep.sh` 自动探测 GPU 计算能力。

## 构建与安装

依赖：Linux、NVIDIA GPU（CUDA ≥ 12）、CMake ≥ 3.24、GCC ≥ 13。

```bash
tar xzf gromacs-2026.3-gpumd-vendor-source-20260818.tar.gz
cd gromacs-2026.3-gpumd-vendor
./build-nep.sh                # 自动探测；CUDA_ARCH=120 为 RTX 5090
export PATH=$(pwd)/build-nep/bin:$PATH
export LD_LIBRARY_PATH=$(pwd)/build-nep/lib:$LD_LIBRARY_PATH
```

## 快速使用

普通 NEP（full-system）：

```text
nnpot-active           = yes
nnpot-modelfile        = nep.txt
nnpot-input-group      = System
nnpot-embedding        = mechanical
```

qNEP 静电完全交由 GROMACS（charge1/charge2 模型，单 rank，全体系）：

```text
nnpot-active           = yes
nnpot-modelfile        = qnep.txt
nnpot-input-group      = System
nnpot-embedding        = mechanical
nnpot-gromacs-charges  = yes
coulombtype            = PME
rcoulomb               = 0.8      ; 要求 = rlist，盒长 ≥ 2.5·rcoulomb
rlist                  = 0.8
pme-order              = 4
```

```bash
gmx grompp -f nvt.mdp -c system.gro -p system.top -o system.tpr
gmx mdrun -s system.tpr -ntmpi 1 -ntomp 1
```

NEP-MM 冒烟测试：

```bash
MODEL=/path/to/nep.txt examples/nep-mm-methane-water/run-test.sh
```

更完整的使用说明见 `docs/NEP-GPUMD.md` 与 `test/` 下的验证脚本。

## 软件版本

- 基于 **GROMACS 2026.3**；
- 内嵌 **GPUMD** 源码（`src/external/gpumd`，与 GPUMD-master 的 NEP/qNEP 力核逐字节一致）；

## 引用要求

使用本软件必须引用以下原始文献：

## 参考文献
**GROMACS**<br>
[1] Spoel D. V. D., Lindahl E., Hess B., et al. GROMACS: fast, flexible, and free [J]. *Journal of Computational Chemistry*, 2005, 26(16): 1701-1718.<br>
[2] Berendsen H. J. C., Vanderspoel D., Vandrunen R. GROMACS: A message-passing parallel molecular dynamics implementation - ScienceDirect [J]. *Computer Physics Communications*, 1995, 91(1-3): 43-56.<br>
[3] Abraham M. J., Murtola T., Schulz R., et al. GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers [J]. *SoftwareX*, 2015, 1-2: 19-25.<br>

**GPUMD and NEP/qNEP**<br>
[4] Fan Z., Wang Y., Ying P., et al. GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations [J]. *The Journal of Chemical Physics*, 2022, 157(11).<br>
[5] Xu K., Bu H., Pan S., et al. GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials [J]. *Materials Genome Engineering Advances*, 2025, 3(3): e70028.<br>
[6] Fan Z., Zeng Z., Zhang C., et al. Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport [J]. *Physical Review B*, 2021, 104(10): 104309.<br>
[7] Fan Z., Tang B., Berger E. E., et al. qNEP: A Highly Efficient Neuroevolution Potential with Dynamic Charges for Large-Scale Atomistic Simulations [J]. *Journal of Chemical Theory and Computation*, 2026, 22(9): 4787-4801.

## 声明（Disclaimer）

本软件按"原样（as-is）"提供，作者不对其准确性、完整性或特定用途的适用性作任何明示或暗示的保证。**所有计算结果（能量、力、结构、动力学性质等）的正确性必须由使用者自行验证**，例如：有限差分力检验、NVE 能量守恒、与原生 GPUMD 的单点/MD 对比、结构函数（RDF）对比等。因使用本软件产生的任何后果由使用者自行承担。模型文件（`nep.txt` 等）的训练与适用性不属于本软件范围。

## 未来可能更新

- qNEP-GROMACS 流级流水线（移除每步设备同步与电荷 D2H，缩小与原生 qNEP ~10% 的速度差距）；
- 多 GPU / domain decomposition（动态电荷全局归约、PME 势跨 rank 分发）；
- 偶极矩输出与 epsilon-surface 修正；OpenCL/SYCL 与 pme-order=5 的势 gather；
- 任意多组动态耦合（pairwise-groups / active-cluster）；
- 动态 NEP-MM 路径优化（邻居表筛选、GPU 端选择、更新周期与滞回、checkpoint）。

---

# GROMACS–GPUMD NEP/qNEP Integration (English Statement)


## Overview

This software integrates the **GPUMD** NEP/qNEP machine-learning potentials into **GROMACS 2026.3**. GPUMD evaluates the NEP/qNEP energies, forces, and virials, while GROMACS retains MD integration, thermostats/barostats, LINCS/SETTLE, classical bonded terms, LJ, Coulomb with GPU PME, trajectories, and energy output. The complete GPUMD source tree is vendored under `src/external/gpumd` and can be upgraded by simple replacement.

## Features

- **Plain NEP**: `nep4`, `nep4_zbl`, `nep5`, `nep5_zbl`; full-system GPU path (~98% of native GPUMD) and static NEP–MM mechanical embedding;
- **qNEP** (`nep4_charge1/2` and `_zbl` variants) in two modes:
  - Native mode: unmodified GPUMD dynamic charges, charge neutrality, and PPPM/Ewald;
  - **GROMACS-owned electrostatics** (`nnpot-gromacs-charges = yes`): qNEP predicts dynamic charges → GROMACS nonbonded/PME charge buffers are updated → GROMACS computes Coulomb energies/forces → GPU PME outputs per-atom reciprocal-space potentials → chain-rule forces `(dE/dq)(dq/dR)`;
- **Dynamic NEP–MM**: adaptive Δ-NEP for two groups (cross-only / a-nep-cross) with automatic active-molecule selection and smooth switching;
- **Fixed-charge long-range scheme** (`nnpot-use-gromacs-electrostatics = yes`) and **energy-offset calibration** (`nnpot-energy-offsets`);
- **GPU-free grompp**: CPU-side model validation and lazy loading — a tpr can be generated without a GPU;
- **One-command build**: `./build-nep.sh` auto-detects the GPU compute capability.

## Build and Install

Requirements: Linux, NVIDIA GPU (CUDA ≥ 12), CMake ≥ 3.24, GCC ≥ 13.

```bash
tar xzf gromacs-2026.3-gpumd-vendor-source-20260818.tar.gz
cd gromacs-2026.3-gpumd-vendor
./build-nep.sh                # auto-detects; CUDA_ARCH=120 for RTX 5090
export PATH=$(pwd)/build-nep/bin:$PATH
export LD_LIBRARY_PATH=$(pwd)/build-nep/lib:$LD_LIBRARY_PATH
```

## Quick Start

Plain NEP (full system):

```text
nnpot-active           = yes
nnpot-modelfile        = nep.txt
nnpot-input-group      = System
nnpot-embedding        = mechanical
```

qNEP with GROMACS-owned electrostatics (charge1/charge2 model, single rank, full system):

```text
nnpot-active           = yes
nnpot-modelfile        = qnep.txt
nnpot-input-group      = System
nnpot-embedding        = mechanical
nnpot-gromacs-charges  = yes
coulombtype            = PME
rcoulomb               = 0.8      ; must equal rlist; box >= 2.5*rcoulomb
rlist                  = 0.8
pme-order              = 4
```

```bash
gmx grompp -f nvt.mdp -c system.gro -p system.top -o system.tpr
gmx mdrun -s system.tpr -ntmpi 1 -ntomp 1
```

NEP-MM smoke test:

```bash
MODEL=/path/to/nep.txt examples/nep-mm-methane-water/run-test.sh
```

See `docs/NEP-GPUMD.md` and the validation scripts under `test/` for details.

## Version

- Based on **GROMACS 2026.3**;
- Vendored **GPUMD** source tree (`src/external/gpumd`; NEP/qNEP force kernels are byte-identical to GPUMD-master);

## Citation Requirement

Users of this software are **required** to cite the following original publications:

**GROMACS**<br>
[1] Spoel D. V. D., Lindahl E., Hess B., et al. GROMACS: fast, flexible, and free [J]. *Journal of Computational Chemistry*, 2005, 26(16): 1701-1718.<br>
[2] Berendsen H. J. C., Vanderspoel D., Vandrunen R. GROMACS: A message-passing parallel molecular dynamics implementation - ScienceDirect [J]. *Computer Physics Communications*, 1995, 91(1-3): 43-56.<br>
[3] Abraham M. J., Murtola T., Schulz R., et al. GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers [J]. *SoftwareX*, 2015, 1-2: 19-25.<br>

**GPUMD and NEP/qNEP**<br>
[4] Fan Z., Wang Y., Ying P., et al. GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations [J]. *The Journal of Chemical Physics*, 2022, 157(11).<br>
[5] Xu K., Bu H., Pan S., et al. GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials [J]. *Materials Genome Engineering Advances*, 2025, 3(3): e70028.<br>
[6] Fan Z., Zeng Z., Zhang C., et al. Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport [J]. *Physical Review B*, 2021, 104(10): 104309.<br>
[7] Fan Z., Tang B., Berger E. E., et al. qNEP: A Highly Efficient Neuroevolution Potential with Dynamic Charges for Large-Scale Atomistic Simulations [J]. *Journal of Chemical Theory and Computation*, 2026, 22(9): 4787-4801.

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of accuracy, completeness, or fitness for a particular purpose. **The correctness of all results (energies, forces, structures, dynamical properties, etc.) must be verified by the users themselves**, e.g. via finite-difference force tests, NVE energy conservation, single-point/MD comparison against native GPUMD, or structural comparisons (RDF). The authors accept no liability for any consequences arising from the use of this software. The training and applicability of the potential files (`nep.txt`, etc.) are outside the scope of this software.

## Possible Future Updates

- Stream-level pipelining of the qNEP-GROMACS mode (remove per-step device synchronization and charge D2H; close the ~10% speed gap to native qNEP);
- Multi-GPU / domain decomposition (global charge reduction, cross-rank PME potential distribution);
- Dipole output and epsilon-surface corrections; potential gather for OpenCL/SYCL and pme-order=5;
- Arbitrary multi-group dynamic coupling (pairwise-groups / active-cluster);
- Dynamic NEP–MM optimizations (neighbor-list screening, GPU-side selection, update intervals and hysteresis, checkpointing).
