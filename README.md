> 当前版本：**GROMACS 2026.4 + GPUMD NEP/qNEP + 可选 WPE 1.0.0 后端**。
> 本次更新内容（中英文）见 **[CHANGELOG.md](CHANGELOG.md)**。
>
> Current release: **GROMACS 2026.4 + GPUMD NEP/qNEP + optional WPE 1.0.0 backend**.
> See **[CHANGELOG.md](CHANGELOG.md)** for the release notes (Chinese and English).

---

# GROMACS 2026.4 + GPUMD NEP/qNEP + WPE 集成版

本目录是 **GROMACS 2026.4** 的源码树，在其上集成了：

- **GPUMD 最新 master**（vendored 于 `src/external/gpumd`，未修改）提供的 NEP4 / qNEP / vdW-NEP 能量、力与 virial；
- **Wisevolve Potential Engine (WPE) 1.0.0** 作为可选的高性能势引擎后端（私有二进制，运行时 `dlopen`）；
- 原有 GROMACS 功能：MD 积分、恒温恒压、LINCS/SETTLE、键角/二面角、LJ、库仑、GPU PME、轨迹与能量输出。

> 状态说明：本版本已完成 2026.4 代码迁移、最新 GPUMD 适配和 WPE 后端接入。
> **GPU 对照验证已完成**（RTX 4090 D）：普通 NEP、qNEP 原生（charge1/charge2）、
> qNEP-GROMACS 链式力、NEP-MM、WPE（PPPM/Ewald）均与 GPUMD 本体逐项对比，
> 能量差 10⁻⁸–10⁻⁵ eV/atom，力 RMS 10⁻⁵–10⁻⁴ eV/Å；WPE 性能约为 GPUMD 的
> 2.09×（plain NEP）/ 2.57×（qNEP，1500 原子）；18k 原子水体系为 2.36×（GROMACS）
> / 2.58×（GPUMD）。同引擎对照下 GROMACS+WPE 仍比 GPUMD+WPE 慢约 7 %（1500）
> / 9.6 %（18k）。已实现设备端事件排序（`GMX_NEP_NO_ASYNC=1` 可回退），收益 1–5 %。
> 详细程序介绍/安装/使用：
> 中文 `docs/程序介绍-安装-使用-中文.md`；
> English `docs/Program-Introduction-Installation-Usage-EN.md`。
> 汇总手册：`docs/完整手册-2026.4.md`；实测明细：`docs/GPU-TESTS-2026.4.md`。

## 1. 后端选择

| 后端 | 说明 | 适用场景 |
|---|---|---|
| `gpumd` | vendored GPUMD CUDA 核（默认回退） | NEP4 / NEP4-ZBL / qNEP charge1-3 / nep_vdw / qnep3_vdw，含 GROMACS 自有静电链式力 |
| `wpe` | Wisevolve Potential Engine 1.0.0（单 GPU，私有运行时） | NEP4 / NEP4-ZBL / qNEP（PPPM 或 Ewald） |
| `auto`（默认） | 若编译了 WPE、模型满足 WPE 形状约束且能找到运行库，则选 WPE；否则 GPUMD | 通用 |

相关 `.mdp` 选项（均为 `nnpot-` 前缀）：

```text
nnpot-backend              = auto | gpumd | wpe
nnpot-wpe-library          = /path/to/wpe.so/lib      ; 可为目录或 .so 文件；不填则查 $GMX_WPE_LIBRARY / $LD_LIBRARY_PATH
nnpot-wpe-kspace           = pppm | ewald             ; WPE 原生 qNEP 倒空间方法
nnpot-wpe-pppm-spacing     = 1.0                      ; Å，0.2–2.0
nnpot-gpumd-kspace         = pppm | ewald             ; GPUMD 原生 qNEP 倒空间方法
nnpot-gpumd-pppm-spacing   = 0                        ; Å，0 表示使用模型默认
nnpot-dftd3                = "pbe 12.0 12.0"          ; 可选 D3 修正（两个后端都支持时）
```

`opt` 之外的既有选项（`nnpot-active`、`nnpot-modelfile`、`nnpot-input-group`、`nnpot-embedding`、`nnpot-gromacs-charges`、`nnpot-charges-nst`、`nnpot-energy-offsets`、`nnpot-adaptive-*` 等）保持兼容。

## 2. 支持的模型

| 模型 | GPUMD 后端 | WPE 后端 | 备注 |
|---|---|---|---|
| `nep4` / `nep4_zbl`（旧格式） | ✅ | ✅ | 旧格式首行标签 |
| `nep <N> ...`（新格式，`format_version 1`） | ✅ | ✅ | 自动转换为 MD 输入 |
| `nep4_charge1/2/3`、`_zbl_`（qNEP） | ✅ 原生 | ✅（qNEP） | WPE 使用自身 PPPM/Ewald |
| qnep1/2/3 新格式 | ✅ | ✅ | |
| `nep_vdw` / `qnep3_vdw`（新格式） | ✅ | ❌ | WPE 1.0.0 未实现 vdW 家族 |
| `nep5` / `nep5_zbl` | ❌ | ❌ | GPUMD 已从 MD 势类中移除 NEP5（仅副本交换工具保留），grompp 明确报错 |
| 温度相关 NEP4 | ❌ | ❌ | 需要温度相关力评估，GROMACS 适配层未驱动 |
| `nep_dipole` / `nep_polarizability` / `nep_spin` | ❌ | ❌ | 非普通能量/力模型，grompp 明确报错 |

WPE 形状约束（在 grompp 阶段静态检查）：`l_max_3body = 4`，ANN 隐藏层宽度 ≤ 256。
WPE 其余限制：只支持一个可见 GPU、单势函数、不支持多势混合、不支持 `dump_observer` / `active` /
TI 系综 / LJ-EAM-Tersoff 的 PIMD。

## 3. qNEP 的两种静电模式

1. **原生模式**（默认）：qNEP 内部完成动态电荷、中性化与 PPPM/Ewald（WPE 或 GPUMD 均可）。
2. **GROMACS 自有静电**（`nnpot-gromacs-charges = yes`）：**仅 GPUMD 后端**。
   qNEP 预测电荷 → 写入 GROMACS 电荷缓冲 → GROMACS 非键 + GPU PME 计算库仑/
   倒空间势 → 链式力 `(dE/dq)(dq/dR)`；要求单 rank、full-system、`pme-order=4`、
   `rcoulomb=rlist`、盒长 ≥ 2.5·`rcoulomb`、不支持 ZBL/charge3/FEP/MTS。
   WPE 后端下该组合会明确报错，而不是静默切换。

## 4. 构建

```bash
cd gromacs-2026.4-nep
CUDA_ARCH=89 ./build-nep.sh                 # RTX 4090 (sm_89)
CUDA_ARCH=120 ./build-nep.sh                # RTX 5090 (sm_120)
WPE=0 CUDA_ARCH=89 ./build-nep.sh           # 不编译 WPE 后端
WPE=1 WPE_LIBRARY=/path/to/wpe.so/lib CUDA_ARCH=89 ./build-nep.sh
BUILD_TYPE=Debug CUDA_ARCH=89 ./build-nep.sh
```

构建脚本默认关闭回归测试下载并使用系统 FFTW；可通过 `BUILD_DIR`、`INSTALL_PREFIX`、
`BUILD_JOBS`、`NATIVE_CPU`、`FAST_MATH`、`LTO`、`RUN_TESTS`、`DO_INSTALL` 覆盖。

WPE **不参与链接**：只有公开 BSD-2 头文件（`src/external/wpe/wisevolve_potential_api.h`）
被 vendor，`gmx` 用 `dlopen()` 在 mdrun 首次求值时加载私有引擎。因此：

- 不安装 WPE 也能构建并运行普通 NEP/qNEP；
- `ldd gmx` 不会出现 `libwisevolve_potential.so`；
- 运行时用 `nnpot-wpe-library`、`GMX_WPE_LIBRARY` 或 `LD_LIBRARY_PATH` 指定引擎位置；
- 不把 WPE 私有二进制复制进源码仓库（许可证见 `src/external/wpe/`）。

## 4.1 更新 GPUMD（保持“直接替换 src/external/gpumd”）

```bash
rm -rf src/external/gpumd
cp -a /path/to/GPUMD-master src/external/gpumd
CUDA_ARCH=89 ./build-nep.sh          # 重新 configure + build
# 或 scripts/update-gpumd-vendor.sh /path/to/GPUMD-master
```

GROMACS 侧不修改 GPUMD 源码；CMake 会校验必需文件、自动纳入新的
`nep_model_file*.cu`、并自动检测可选的 vdW 模型族。若上游 API 变化，
`docs/GPUMD-UPGRADE.md` 列出了适配层依赖的完整接口清单与处理方式。
本轮已用“删除后换成原始 GPUMD-master 解压树”的方式实测替换并构建成功。

## 4.2 grompp 不需要 GPU

`gmx grompp` 只做 CPU 端模型解析/校验、拓扑预处理与后端选择：

- 不初始化 CUDA，不打开 `/dev/nvidia*`，不 `dlopen` WPE；
- `CUDA_VISIBLE_DEVICES=""`（无可见 GPU）下可正常生成 tpr；
- 与有 GPU 时生成的 tpr 内容一致（仅随机种子/时间等元数据不同）；
- 生成的 tpr 可拷贝到 GPU 服务器直接 `gmx mdrun`（含 `nnpot-backend = wpe`）。

验证：

```bash
CUDA_VISIBLE_DEVICES="" gmx grompp -f nvt.mdp -c sys.gro -p sys.top -o cpu.tpr
CUDA_VISIBLE_DEVICES="" LD_DEBUG=libs gmx grompp ... 2>&1 | grep wisevolve   # 无输出
strace -f -e openat gmx grompp ... 2>&1 | grep /dev/nvidia                     # 无输出
```

## 5. 运行

```bash
export LD_LIBRARY_PATH=$PWD/build-nep/lib:$LD_LIBRARY_PATH   # 仅在使用构建树自带的 libgromacs 时需要

# 普通 NEP / qNEP（GPUMD）
gmx grompp -f nvt.mdp -c sys.gro -p sys.top -o sys.tpr
gmx mdrun -s sys.tpr -ntmpi 1 -ntomp 1

# WPE
export GMX_WPE_LIBRARY=/path/to/wpe.so/lib
# mdp 中可显式 nnpot-backend = wpe
gmx mdrun -s sys.tpr -ntmpi 1 -ntomp 1
```

免显卡 grompp、模型校验、后端选择均在 CPU 完成；WPE 运行库只在 mdrun 首次求值时加载。

## 6. 验证

```bash
# 单元测试（含 NEP 模型解析 / WPE 形状检查）
ctest --test-dir build-nep -R NNPotAppliedForcesUnitTest --output-on-failure

# CPU 侧端到端验证（16 项）
GMX=$PWD/build-nep/bin/gmx WPE_LIB=/path/to/wpe.so/lib \
  test/2026.4-validation/run_cpu_validation.sh

# WPE 公开 ABI 探测（无需 GPU）
python3 test/2026.4-validation/wpe_abi_probe.py /path/to/wpe.so/lib/libwisevolve_potential.so
```

GPU 数值验证（有限差分、NVE 漂移、与原生 GPUMD/WPE 单点对比、系综与性能）已在
RTX 4090 D 上执行，数据见 `docs/GPU-TESTS-2026.4.md` 与 `docs/完整手册-2026.4.md`；
脚本为 `test/2026.4-validation/gpu/run_gpu_validation.sh`。

## 7. 已知限制与差异

- **NEP5 不再支持**：最新 GPUMD 的 MD 势类只保留 NEP4；旧版 GROMACS-NEP 的
  `nep5` 路径随之上移移除（grompp 给出原因）。
- **WPE 与 GROMACS 自有静电不兼容**：WPE 在引擎内部完成 qNEP 静电，无法输出
  GPUMD 的 `D_real` 链式势；该组合会报错并建议 `nnpot-backend = gpumd`。
- **WPE 单 GPU**：WPE 只使用一个可见 GPU；GROMACS 侧的 NEP 设备路径仍要求单 rank、
  无 DD（与 2026.3 版一致）。
- **qNEP-GROMACS 小盒限制**：仍要求盒长 ≥ 2.5·`rcoulomb`（沿用旧版保守限制）。
- **WPE 二进制依赖**：发布包基于 CUDA 12 构建，需要 `libcufft.so.11`；CUDA 13 系统
  需要 Wisevolve 提供对应构建。加载失败时 mdrun 会报告缺失的库名。
- GPU/WPE 数值与性能已在 RTX 4090 D 上验证；WPE 1.0.0 需要 CUDA 12 的
  `libcufft.so.11`，CUDA 13-only 环境需 Wisevolve 的 CUDA 13 构建。
- 当前 NEP/WPE 设备路径仍要求单 rank、无 DD；单 GPU 18k 为已验证的最大规模。

## 8. 引用与许可

- GROMACS: https://www.gromacs.org （LGPL-2.1+）
- GPUMD: https://github.com/brucefan1983/GPUMD （LGPL-3.0+，见 `src/external/gpumd/LICENSES`）
- WPE: 公开接口 BSD-2-Clause；引擎为 Wisevolve 专有软件，见
  `src/external/wpe/LICENSE-WPE` / `NOTICE`，引用：
  Hekai Bu and Zheyong Fan, arXiv:2610.17366 (2026).
