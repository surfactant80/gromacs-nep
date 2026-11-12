# 更新说明 / Changelog

本文件记录 **GROMACS–GPUMD NEP/qNEP 集成版** 的版本更新。中文说明在前，English release
notes 在后（两节内容对应）。发行物对应的源码包为 `gromacs-2026.4-nep.zip`。

This file records the release history of the **GROMACS–GPUMD NEP/qNEP integration**.
The Chinese notes come first and the English notes follow (both sections cover the
same changes). The corresponding source archive is `gromacs-2026.4-nep.zip`.

---

# 一、中文更新说明

## v2026.4（2026-11-12）

**基线：GROMACS 2026.4（上游发布于 2026-10-02）＋ 当前 GPUMD master ＋ 可选 WPE 1.0.0**

这是本项目的第二次公开发布，替代此前的 GROMACS 2026.3 基线版本。相对于上一版，
本次是**功能级更新**：宿主从 2026.3 升级到 2026.4，内嵌 GPUMD 引擎更新到上游最新
master，并新增了 WPE（Wisevolve Potential Engine）高性能后端与 `qnep-charge`
轨迹分析工具。

### 1. 变更规模一览

| 项目 | 数量 |
|---|---:|
| 变更文件总数 | 1483（新增 690 / 修改 572 / 删除 217 / 类型变更 4） |
| 其中 `src/external/gpumd` | 1223（上游引擎更新） |
| 其中 GROMACS 手册页 `docs/man` | 102（随上游 2026.4 重新生成） |
| 其中 `src/gromacs` 源码 | 50（含 `nnpot` 20 项） |
| 其中 `examples/` | 48（全部为误提交的运算输出/备份文件，已删除） |
| 其中构建与脚本 | 9 |
| 其中文档 | 8 |
| 其中新增 `src/external/wpe` | 4（公开 ABI 头文件与许可） |

> 结论：NEP 适配层本身（`src/gromacs/applied_forces/nnpot`）只做了小幅扩展，
> 绝大部分变化来自上游 GROMACS 2026.4 与上游 GPUMD 的同步。

### 2. 宿主升级：GROMACS 2026.3 → 2026.4

2026.4 是上游的缺陷修正版本（bugfix release），不含新功能。与本项目使用场景最相关的
修正包括：

- **NNP 对列表的 cell shift 修正**：对以 GROMACS 对列表作为输入的 NNP 模型，三斜盒
  下的 cell shift 计算错误会导致能量与力错误（上游 issue 5705）。本项目 NEP 路径走
  full-system 坐标直通，不经过该对列表，但该修正确保了 `nnpot` 模块与上游一致。
- **多 PME rank 下的 PME 堆缓冲区溢出**：z 方向需要 SIMD padding 时，major 维重叠
  通信可能越界写入，`pme-order = 5` 时最易触发（issue 5669）。
- **分阶段 PP–PME 通信下的输出力错误**：GPU + 独立 PME rank + staged 通信时，非
  维里步写出的力缺少 PME 贡献。
- **GPU 非局部非键 kernel 使用上一步 shift 向量的竞态**：压力耦合（盒子变化）时
  可能导致跨周期边界力错误（issue 5724）。
- **MPI 并行 `mdrun -plumed` 启动即中止**（issue 5671，2026 系列特有）。
- **SYCL PME 在 oneAPI 2026.x 上的误编译规避**（issue 5688）。
- 其它：`grompp` 大 ifdef 块栈溢出、`gmx eneconv` 时间戳精度、`gmx awh` 摩擦因子
  的 kT² 换算、FLOPs 版动态负载均衡失效、`gmx nonbonded-benchmark` 的 CSV 输出、
  不再安装 lmfit 内容、AMBER14SB/19SB 的 DOI 更新、Colvars 库修复。

**对本项目的影响**：NEP/qNEP 的数值路径没有改变；但由于 tpr 与宿主版本绑定，
**升级后必须重新 `gmx grompp`**，不能沿用 2026.3 生成的 tpr。

### 3. 引擎升级：内嵌 GPUMD 更新到当前 master

`src/external/gpumd` 依旧是**未经修改**的上游源码树（可直接整体替换），但内容已同步到
上游最新 master（2026-11），对应 1223 个文件的新增/修改/删除。用户可见的变化：

- **NEP5 从 MD 势类中移除**：上游 MD 势类只保留 NEP4 与新格式。本项目的 `nep5` /
  `nep5_zbl` 支持随之移除，`grompp` 会给出明确报错，而不是静默失败。
  （GPUMD 的副本交换工具与训练工具仍保留 NEP5 相关代码，那不参与本项目的 MD 路径。）
- **新增 vdW-NEP 模型族**：`nep_vdw`、`qnep3_vdw`（GPUMD 后端支持；WPE 1.0.0 未实现）。
- **qNEP 扩展**：`nep4_charge1/2/3` 及其 `_zbl` 变体、新格式 `qnep1/2/3` 全部支持。
- **新模型格式**：支持 `nep <N> <elements>`（`format_version 1`）格式，可与旧格式
  （`nep4 ...`）混用，解析在 CPU 端完成。
- **倒空间方法可选**：GPUMD 原生 qNEP 的 PPPM/Ewald 现在可通过
  `nnpot-gpumd-kspace`、`nnpot-gpumd-pppm-spacing` 选择。
- **D3 色散修正**：新增 `nnpot-dftd3`（形如 `"pbe 12.0 12.0"`），在模型与后端都
  支持时生效。
- **无 `run.in` 依赖**：GPUMD 构造所需的 `RunInput` 由 mdp 选项在内存中合成，
  不再读取或生成 `run.in` 文件。
- 上游清理：删除 167 个上游已移除的示例/文档文件，vendor 树更精简。

替换流程仍然是一行：`rm -rf src/external/gpumd && cp -a /path/to/GPUMD-master src/external/gpumd`，
或直接使用新增的 `scripts/update-gpumd-vendor.sh`。

### 4. 新增功能：WPE（Wisevolve Potential Engine 1.0.0）后端

本次最重要的新特性。WPE 是 Wisevolve 提供的闭源高性能 NEP/qNEP 推理引擎，作为
**可选后端**接入，与内嵌 GPUMD 后端并存：

- **后端选择**：新增 `nnpot-backend = auto | gpumd | wpe`。默认 `auto`：若编译了
  WPE、模型满足 WPE 形状约束且能找到运行库，则选 WPE；否则回退 GPUMD，并在
  grompp/md.log 中说明原因。后端的静态校验全部在 grompp 阶段完成。
- **不参与链接**：源码树只包含公开 BSD-2 许可的头文件
  `src/external/wpe/wisevolve_potential_api.h`；`gmx` 在 `mdrun` 首次求值时用
  `dlopen()` 加载私有引擎，因此 `ldd gmx` 不会出现 `libwisevolve_potential.so`，
  没有 WPE 也能构建和运行普通 NEP/qNEP。
- **运行库定位**：`nnpot-wpe-library`（可给 `.so` 或所在目录）、`GMX_WPE_LIBRARY`
  环境变量、或 `LD_LIBRARY_PATH`。
- **倒空间选项**：`nnpot-wpe-kspace = pppm | ewald`、
  `nnpot-wpe-pppm-spacing`（Å，0.2–2.0，默认 1.0）。
- **形状约束**（grompp 静态检查）：`l_max_3body = 4`，ANN 隐藏层宽度 ≤ 256。
- **限制**：只使用一个可见 GPU；不支持 vdW 模型族；与
  `nnpot-gromacs-charges`（qNEP 交由 GROMACS 算静电）互斥，该组合会显式报错并
  建议改用 `nnpot-backend = gpumd`。
- **依赖**：WPE 1.0.0 发布包基于 CUDA 12 构建，需要 `libcufft.so.11`；纯 CUDA 13
  环境需向 Wisevolve 索取 CUDA 13 构建。加载失败时 `mdrun` 会报告缺失的库名。
- **性能收益**（RTX 4090 D，单 GPU，NVE，0.5 fs，与 GPUMD 本体对比）：
  1500 原子 **2.09×（plain NEP）/ 2.57×（qNEP）**；18 000 原子 **2.36×（GROMACS
  侧）/ 2.58×（GPUMD 侧）**。同引擎对照下 GROMACS+WPE 仍比 GPUMD+WPE 慢约
  7%（1500 原子）/ 9.6%（18k 原子）。

### 5. 新增工具：`gmx qnep-charge`

新的轨迹分析模块，用于在**不启动外部进程**的前提下复算 qNEP 动态电荷：

- 输入：轨迹 + 拓扑 + GPUMD qNEP 模型文件；
- 在 GPU 上调用内嵌 GPUMD 核逐帧预测每原子电荷，输出所选原子电荷随时间的变化；
- 可选 `-dipole` 同时输出逐帧偶极矩。

### 6. 适配层与运行时改进

- **异步设备端事件排序**：NEP/WPE 的 kernel 与 GROMACS 力归约通过设备端事件排序，
  非能量步不再每步做 host 同步；收益约 1–5%。可用 `GMX_NEP_NO_ASYNC=1` 回退到
  逐步同步路径（对比/排障）。
- **后端相关环境变量**：`GMX_WPE_LIBRARY`、`GMX_WPE_ATOM_VIRIAL=1`（请求逐原子
  virial，调试用）、`GMX_QNEP_DEBUG`、`GMX_CUDA_GRAPH`、`GMX_NEP_NO_ASYNC`。
- **免 GPU 的 grompp 保持并加强**：模型解析/校验迁移到纯 CPU 实现
  （`nepmodelinfo.cpp`），grompp 不初始化 CUDA、不打开 `/dev/nvidia*`、不 `dlopen`
  WPE；纯 CPU 机器生成的 tpr 可拷到 GPU 服务器直接 `mdrun`（包括
  `nnpot-backend = wpe`）。
- **qNEP-GROMACS 链式力模式保留**，限制不变：单 rank、full-system、
  `pme-order = 4`、`rcoulomb = rlist`、盒长 ≥ 2.5·`rcoulomb`；不支持 ZBL /
  charge3 / FEP / MTS。

### 7. 构建系统

`build-nep.sh` 已重写，覆盖 WPE 与更多构建参数：

```bash
CUDA_ARCH=89 ./build-nep.sh                 # RTX 4090 (sm_89)
CUDA_ARCH=120 ./build-nep.sh                # RTX 5090 (sm_120)
WPE=0 CUDA_ARCH=89 ./build-nep.sh           # 不编译 WPE 后端
WPE=1 WPE_LIBRARY=/path/to/wpe.so/lib CUDA_ARCH=89 ./build-nep.sh
LTO=1 CLEAN_BUILD=1 CUDA_ARCH=89 ./build-nep.sh
DO_INSTALL=1 CUDA_ARCH=89 ./build-nep.sh
```

- 新增 `WPE` / `WPE_LIBRARY` 开关，对应 CMake 选项 `GMX_NEP_WPE=ON|OFF`；
- 新增 `LTO`、`FAST_MATH`、`CLEAN_BUILD`、`RUN_TESTS`、`DO_INSTALL`；
- CUDA 架构优先从 `nvidia-smi` 自动探测，失败时提示显式设置 `CUDA_ARCH`；
- 构建脚本默认关闭回归测试下载、使用系统 FFTW3、AVX2_256、thread-MPI；
- 修订后的环境基线：Linux x86-64、CMake ≥ 3.28、GCC 15.x、CUDA 13.x
  （实测 13.4 / nvcc 13.4.92）、支持 AVX2 的 CPU、sm_80 及以上 NVIDIA GPU
  （实测 RTX 4090 D，sm_89，24 GB）。

### 8. 文档

- 新增**中文**《程序介绍、安装与使用》`docs/程序介绍-安装-使用-中文.md`（552 行，
  含架构、安装、mdp 选项、模型支持、精度/性能、限制、故障排查）。
- 新增**英文**同名文档 `docs/Program-Introduction-Installation-Usage-EN.md`。
- 重写 `docs/NEP-GPUMD.md`：反映 2026.4 状态（NEP4-only、后端选择、新格式、
  WPE、电荷读取、自适应耦合）。
- 保留 `NEP-INTEGRATION-REPORT.md` 作为 2026.3 轮次的历史报告（文首已标注）。
- 新增 `scripts/update-gpumd-vendor.sh`，封装“删除后整体替换 GPUMD”的流程。

### 9. 验证

- 单元测试：`NNPotAppliedForcesUnitTest` **1/1 通过**（含模型解析、WPE 形状检查）。
- CPU 端到端：`test/2026.4-validation/run_cpu_validation.sh` **16/16 通过**
  （含无 GPU 的 grompp、后端选择、WPE 缺失库报错、纯 MM 对照）。
- GPU 数值验证（RTX 4090 D）：有限差分最大相对误差 **0.95%**；水体系 qNEP NVE
  漂移 **−3.1×10⁻⁶ eV/(atom·ps)**（异步路径 −2.8×10⁻⁶）。
- 单点精度（对 GPUMD 本体）：普通 NEP ΔE 1.7e-5 eV/atom、力 RMS 5.1e-5 eV/Å；
  qNEP 原生 ~5e-6 / 3.5–3.9e-5；qNEP+PME −6.8e-6 / 1.9e-4；
  WPE PPPM −2.8e-8 / 1.0e-5；WPE Ewald −8.8e-8 / 1.0e-5。
- 系综：NPT/NPH 与 GPUMD 的压力差 **1.0–7.5 bar（≤0.8σ）**，在统计误差内。
- 新增 `test/2026.4-validation/wpe_abi_probe.py`（无需 GPU 探测 WPE 公开 ABI）
  与 `test/2026.4-validation/gpu/` 下的 GPU 验证脚本。

### 10. 仓库整理

- 删除 `examples/nep-mm-methane-water/work/` 下 48 个误提交的运行产物与编辑器备份
  （`#*#`、`.trr`、`.edr`、`.tpr`、`.log`、`.gro`、`.xvg` 等）。示例只保留输入与
  脚本，运行后自行生成输出。
- 保留仓库级 `LICENSE`（MIT）与 `README.md`，并把 `README.md` 更新为 2026.4 内容。
- `.vscode/` 仍按上游 GPUMD 的 `.gitignore` 忽略（不纳入版本库）。
- `CITATION.cff`、`CMakeLists.txt`、`INSTALL`、`INSTALL-NEP.md`、`admin/` 与
  `cmake/` 随上游版本同步更新。

### 11. 破坏性变更与迁移

1. **NEP5 不再支持**：`nep5` / `nep5_zbl` 模型在 `grompp` 阶段报错。请改用 NEP4
   或新格式模型（上游提供 `tools/Miscellaneous/for_coding/nep4nep5/` 的转换脚本，
   但注意 NEP5→NEP4 不是无损替换，需要重新训练/验证）。
2. **必须重新生成 tpr**：宿主从 2026.3 升到 2026.4，旧 tpr 不兼容。
3. **mdp 选项**：新增选项均有默认值（`nnpot-backend = auto` 等），旧的 2026.3
   输入文件无需修改即可运行；但若显式启用 WPE，需要提供运行库。
4. **WPE 与 GROMACS 静电互斥**：`nnpot-backend = wpe` +
   `nnpot-gromacs-charges = yes` 会在 grompp 阶段被拒绝。
5. **构建参数**：`GMX_NEP_WPE` 为新增 CMake 选项，默认由 `build-nep.sh` 的 `WPE`
   变量控制（默认 ON）。手工 CMake 时若系统没有 WPE 头文件，需显式加
   `-DGMX_NEP_WPE=OFF`。

### 12. 已知限制

- NEP/WPE 设备路径要求单 rank、无 domain decomposition；WPE 只用单 GPU；
- qNEP-GROMACS 小盒限制（盒长 ≥ 2.5·`rcoulomb`）沿用；
- `nep_dipole` / `nep_polarizability` / `nep_spin` 与温度相关 NEP4 不支持
  （非普通能量/力模型，grompp 明确报错）；
- 全体系 NEP 下 GROMACS 关闭经典非键/PME，`nnpot-charges-nst` 的二进制电荷输出
  不触发（tpr 中仍含 `QNC1` 头）；
- 已验证最大规模 18 000 原子；更大体系、多 GPU、DD 未测试；
- 性能数字基于 0.5 fs 步长，使用 HMR/SETTLE 放大 dt 时需重新验证。

### 13. 发行物校验

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `gromacs-2026.4-nep.zip` | 83 090 172 B | `B51D24959B9A3505791E1ECAF66865E59F5338EB933D6B26FFC68D8254456C6B` |

### 14. 引用

使用本软件需要引用 GROMACS、GPUMD、NEP、qNEP 的原始文献；使用 WPE 时还需引用
Hekai Bu and Zheyong Fan, arXiv:2610.17366 (2026)。完整列表见 `README-NEP.md`。

---

# 二、English Release Notes

## v2026.4 (2026-11-12)

**Baseline: GROMACS 2026.4 (upstream release 2026-10-02) + current GPUMD master + optional WPE 1.0.0**

This is the second public release of the project and supersedes the previous
GROMACS 2026.3-based version. It is a feature-level update: the host is bumped
from 2026.3 to 2026.4, the vendored GPUMD engine is synchronized to current
upstream master, and a new optional high-performance backend, WPE (Wisevolve
Potential Engine), plus a `qnep-charge` trajectory-analysis tool are added.

### 1. Change statistics

| Item | Count |
|---|---:|
| Total changed files | 1483 (690 added / 572 modified / 217 deleted / 4 type changes) |
| of which `src/external/gpumd` | 1223 (upstream engine update) |
| of which `docs/man` | 102 (regenerated with upstream 2026.4) |
| of which `src/gromacs` sources | 50 (including 20 in `nnpot`) |
| of which `examples/` | 48 (all accidentally committed run outputs/backups; removed) |
| of which build and scripts | 9 |
| of which documentation | 8 |
| of which new `src/external/wpe` | 4 (public ABI header and license) |

In short: the NEP adapter itself (`src/gromacs/applied_forces/nnpot`) only grew
modestly; almost all churn comes from the upstream GROMACS 2026.4 and upstream
GPUMD synchronization.

### 2. Host upgrade: GROMACS 2026.3 → 2026.4

2026.4 is an upstream bugfix release with no new features. The fixes most
relevant to this project:

- **NNP pair-list cell shifts**: for NNP models that request the GROMACS pair
  list, cell shifts were computed incorrectly, giving wrong energies and forces
  on triclinic boxes (upstream issue 5705). The NEP path here streams full-system
  coordinates and does not consume that pair list, but the fix keeps the `nnpot`
  module aligned with upstream.
- **PME heap buffer overflow with multiple PME ranks** when SIMD padding was
  needed in z; most readily triggered with `pme-order = 5` (issue 5669).
- **Wrong output forces with staged PP–PME communication** (GPU, separate PME
  rank): forces written on non-virial steps lacked the PME contribution.
- **Race on stale shift vectors** in the GPU non-local nonbonded kernel with a
  changing box (pressure coupling), issue 5724.
- **MPI-parallel `mdrun -plumed` aborting at startup** (issue 5671, 2026-only).
- **SYCL PME miscompilation workaround** for oneAPI 2026.x (issue 5688).
- Also: `grompp` stack overflow with large ifdef blocks, `gmx eneconv` time
  precision, `gmx awh` friction off by kT², FLOPs-based dynamic load balancing,
  `gmx nonbonded-benchmark` CSV output, lmfit content no longer installed,
  AMBER14SB/19SB DOI update, and Colvars fixes.

**Impact here**: the NEP/qNEP numerical path is unchanged, but tpr files are
tied to the host version, so **you must re-run `gmx grompp`**; 2026.3 tpr files
are not reusable.

### 3. Engine update: vendored GPUMD synchronized to current master

`src/external/gpumd` remains an **unmodified** upstream source tree (replace it
wholesale to upgrade), now synced to current upstream master (2026-11): 1223
files added/modified/deleted. User-visible changes:

- **NEP5 dropped from the MD potential classes**: upstream MD potentials now
  implement NEP4 and the new model format only. `nep5` / `nep5_zbl` support is
  therefore removed here, with an explicit `grompp` error instead of a silent
  failure. (GPUMD's replica-exchange and training tools still carry NEP5 code;
  that is not part of the MD path used here.)
- **New vdW-NEP model family**: `nep_vdw`, `qnep3_vdw` (GPUMD backend; not
  implemented in WPE 1.0.0).
- **qNEP extensions**: `nep4_charge1/2/3` and their `_zbl` variants, plus the new
  format `qnep1/2/3`, are all supported.
- **New model format**: `nep <N> <elements>` (`format_version 1`) alongside the
  legacy `nep4 ...` layout; parsing happens on the CPU side.
- **Selectable reciprocal-space method** for native GPUMD qNEP through
  `nnpot-gpumd-kspace` and `nnpot-gpumd-pppm-spacing`.
- **D3 dispersion correction**: new `nnpot-dftd3` option (e.g. `"pbe 12.0 12.0"`),
  applied when both the model and the backend support it.
- **No `run.in` dependency**: the `RunInput` required by GPUMD constructors is
  synthesized in memory from mdp options; no `run.in` is read or written.
- Upstream cleanup: 167 obsolete example/documentation files removed; the vendor
  tree is leaner.

Upgrading stays a one-liner:
`rm -rf src/external/gpumd && cp -a /path/to/GPUMD-master src/external/gpumd`,
or use the new `scripts/update-gpumd-vendor.sh`.

### 4. New feature: WPE (Wisevolve Potential Engine 1.0.0) backend

The headline addition. WPE is a closed-source, high-performance NEP/qNEP
inference engine from Wisevolve, integrated as an **optional backend** next to
the vendored GPUMD one:

- **Backend selection**: new `nnpot-backend = auto | gpumd | wpe`. The default
  `auto` picks WPE when it was compiled in, the model satisfies the WPE shape
  constraints, and the runtime library can be found; otherwise it falls back to
  GPUMD and reports the reason in grompp/`md.log`. All backend validation is
  static and happens at grompp time.
- **Never linked**: the source tree ships only the public BSD-2 licensed header
  `src/external/wpe/wisevolve_potential_api.h`. `gmx` `dlopen()`s the private
  engine on first force evaluation in `mdrun`, so `ldd gmx` never shows
  `libwisevolve_potential.so`, and builds without WPE still run plain NEP/qNEP.
- **Runtime lookup**: `nnpot-wpe-library` (a `.so` file or its directory),
  the `GMX_WPE_LIBRARY` environment variable, or `LD_LIBRARY_PATH`.
- **Reciprocal-space options**: `nnpot-wpe-kspace = pppm | ewald` and
  `nnpot-wpe-pppm-spacing` (Å, 0.2–2.0, default 1.0).
- **Shape constraints** (static checks in grompp): `l_max_3body = 4`, ANN hidden
  layer width ≤ 256.
- **Limitations**: single visible GPU only; the vdW model family is not
  implemented; incompatible with `nnpot-gromacs-charges` (qNEP electrostatics
  owned by GROMACS) — that combination errors out and recommends
  `nnpot-backend = gpumd`.
- **Dependency**: the WPE 1.0.0 distribution is built against CUDA 12 and needs
  `libcufft.so.11`; CUDA 13-only environments need a CUDA 13 build from
  Wisevolve. Load failures name the missing library.
- **Performance** (RTX 4090 D, single GPU, NVE, 0.5 fs, versus native GPUMD):
  1500 atoms **2.09× (plain NEP) / 2.57× (qNEP)**; 18 000 atoms **2.36×
  (GROMACS side) / 2.58× (GPUMD side)**. Engine-for-engine, GROMACS+WPE is
  still ~7% (1500 atoms) / ~9.6% (18k atoms) slower than GPUMD+WPE.

### 5. New tool: `gmx qnep-charge`

A new trajectory-analysis module that recomputes qNEP dynamic charges **without
an external process**:

- Inputs: a trajectory, a topology, and a GPUMD qNEP model file;
- runs the vendored GPUMD kernels on the GPU to predict per-atom charges for
  every frame and writes the selected atoms' charges over time;
- optional `-dipole` also writes per-frame dipole moments.

### 6. Adapter and runtime improvements

- **Asynchronous device-side event ordering** between the NEP/WPE kernels and
  the GROMACS force reduction removes the per-step host synchronization on
  non-energy steps; gain ~1–5%. `GMX_NEP_NO_ASYNC=1` restores the
  synchronize-every-step path for comparison and debugging.
- **Backend-related environment variables**: `GMX_WPE_LIBRARY`,
  `GMX_WPE_ATOM_VIRIAL=1` (per-atom virial, debugging), `GMX_QNEP_DEBUG`,
  `GMX_CUDA_GRAPH`, `GMX_NEP_NO_ASYNC`.
- **GPU-free grompp preserved and strengthened**: model parsing/validation moved
  to a CPU-only implementation (`nepmodelinfo.cpp`); grompp does not initialize
  CUDA, does not open `/dev/nvidia*`, and does not `dlopen` WPE. A tpr generated
  on a CPU-only machine (including `nnpot-backend = wpe`) can be copied to a
  GPU node and run directly.
- **qNEP-GROMACS chain-rule mode retained** under the same restrictions: single
  rank, full system, `pme-order = 4`, `rcoulomb = rlist`, box ≥ 2.5·`rcoulomb`;
  no ZBL / charge3 / FEP / MTS.

### 7. Build system

`build-nep.sh` has been rewritten to cover WPE and more build knobs:

```bash
CUDA_ARCH=89 ./build-nep.sh                 # RTX 4090 (sm_89)
CUDA_ARCH=120 ./build-nep.sh                # RTX 5090 (sm_120)
WPE=0 CUDA_ARCH=89 ./build-nep.sh           # build without the WPE backend
WPE=1 WPE_LIBRARY=/path/to/wpe.so/lib CUDA_ARCH=89 ./build-nep.sh
LTO=1 CLEAN_BUILD=1 CUDA_ARCH=89 ./build-nep.sh
DO_INSTALL=1 CUDA_ARCH=89 ./build-nep.sh
```

- New `WPE` / `WPE_LIBRARY` switches, mapping to the CMake option
  `GMX_NEP_WPE=ON|OFF`;
- New `LTO`, `FAST_MATH`, `CLEAN_BUILD`, `RUN_TESTS`, `DO_INSTALL`;
- CUDA architecture is auto-detected from `nvidia-smi`, with a clear message
  asking for an explicit `CUDA_ARCH` when detection fails;
- regression-test download off, system FFTW3, AVX2_256, thread-MPI by default;
- revised baseline: Linux x86-64, CMake ≥ 3.28, GCC 15.x, CUDA 13.x (measured
  13.4 / nvcc 13.4.92), AVX2-capable CPU, NVIDIA GPU sm_80 or newer (measured
  RTX 4090 D, sm_89, 24 GB).

### 8. Documentation

- New Chinese manual `docs/程序介绍-安装-使用-中文.md` (552 lines: architecture,
  installation, mdp options, model support, accuracy/performance, limitations,
  troubleshooting).
- New English counterpart `docs/Program-Introduction-Installation-Usage-EN.md`.
- `docs/NEP-GPUMD.md` rewritten for the 2026.4 state (NEP4-only, backend
  selection, new format, WPE, charge access, adaptive coupling).
- `NEP-INTEGRATION-REPORT.md` kept as the historical 2026.3 report (now marked
  as such at the top).
- New `scripts/update-gpumd-vendor.sh` wrapping the "delete and replace GPUMD"
  workflow.

### 9. Validation

- Unit test: `NNPotAppliedForcesUnitTest` **1/1 passed** (model parsing, WPE
  shape checks).
- CPU end to end: `test/2026.4-validation/run_cpu_validation.sh` **16/16
  passed** (GPU-free grompp, backend selection, missing-WPE-library error, plain
  MM control).
- GPU numerics (RTX 4090 D): finite-difference maximum relative error **0.95%**;
  water qNEP NVE drift **−3.1×10⁻⁶ eV/(atom·ps)** (−2.8×10⁻⁶ on the async path).
- Single-point accuracy versus native GPUMD: plain NEP ΔE 1.7e-5 eV/atom, force
  RMS 5.1e-5 eV/Å; native qNEP ~5e-6 / 3.5–3.9e-5; qNEP+PME −6.8e-6 / 1.9e-4;
  WPE PPPM −2.8e-8 / 1.0e-5; WPE Ewald −8.8e-8 / 1.0e-5.
- Ensembles: NPT/NPH pressure difference versus GPUMD **1.0–7.5 bar (≤0.8σ)**,
  within statistical error.
- New `test/2026.4-validation/wpe_abi_probe.py` (probe the public WPE ABI
  without a GPU) and GPU validation scripts under `test/2026.4-validation/gpu/`.

### 10. Repository housekeeping

- Removed 48 accidentally committed run outputs and editor backups under
  `examples/nep-mm-methane-water/work/` (`#*#`, `.trr`, `.edr`, `.tpr`, `.log`,
  `.gro`, `.xvg`, ...). Examples now ship inputs and scripts only; outputs are
  produced by running them.
- Kept the repository-level `LICENSE` (MIT) and `README.md`, with `README.md`
  updated to the 2026.4 content.
- `.vscode/` stays ignored by upstream GPUMD's `.gitignore` (not tracked).
- `CITATION.cff`, `CMakeLists.txt`, `INSTALL`, `INSTALL-NEP.md`, `admin/` and
  `cmake/` synced with the upstream version.

### 11. Breaking changes and migration

1. **NEP5 is no longer supported**: `nep5` / `nep5_zbl` models fail at `grompp`.
   Move to NEP4 or the new model format (upstream ships conversion helpers under
   `tools/Miscellaneous/for_coding/nep4nep5/`; a NEP5→NEP4 conversion is not
   lossless, so retrain and revalidate).
2. **Regenerate tpr files**: the host moved from 2026.3 to 2026.4; old tpr files
   are incompatible.
3. **mdp options**: all new options have defaults (`nnpot-backend = auto`, ...),
   so existing 2026.3 inputs run unchanged; explicitly selecting WPE requires
   the runtime library.
4. **WPE and GROMACS-owned electrostatics are mutually exclusive**:
   `nnpot-backend = wpe` with `nnpot-gromacs-charges = yes` is rejected by
   grompp.
5. **Build option**: `GMX_NEP_WPE` is new; `build-nep.sh` drives it through `WPE`
   (default ON). With hand-written CMake and no WPE headers available, pass
   `-DGMX_NEP_WPE=OFF` explicitly.

### 12. Known limitations

- NEP/WPE device paths require a single rank and no domain decomposition; WPE
  uses one GPU only;
- the qNEP-GROMACS small-box restriction (box ≥ 2.5·`rcoulomb`) is unchanged;
- `nep_dipole` / `nep_polarizability` / `nep_spin` and temperature-dependent
  NEP4 are unsupported (not plain energy/force models; grompp errors out);
- for full-system NEP, GROMACS disables classical nonbonded/PME, so the
  `nnpot-charges-nst` binary charge output is not triggered (the tpr still
  carries the `QNC1` header);
- the largest validated system is 18 000 atoms; larger systems, multi-GPU and
  domain decomposition are untested;
- performance figures use a 0.5 fs timestep; using HMR/SETTLE to increase dt
  requires re-validating dynamics and thermodynamics.

### 13. Release artifact checksum

| File | Size | SHA-256 |
|---|---:|---|
| `gromacs-2026.4-nep.zip` | 83,090,172 B | `B51D24959B9A3505791E1ECAF66865E59F5338EB933D6B26FFC68D8254456C6B` |

### 14. Citation

Using this software requires citing the original GROMACS, GPUMD, NEP and qNEP
publications; using WPE additionally requires Hekai Bu and Zheyong Fan,
arXiv:2610.17366 (2026). See `README-NEP.md` for the complete list.
