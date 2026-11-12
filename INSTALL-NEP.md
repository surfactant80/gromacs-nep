# 安装与构建（GROMACS 2026.4 + GPUMD NEP + WPE）

## 1. 依赖

| 组件 | 版本（本仓库验证） | 说明 |
|---|---|---|
| 操作系统 | Linux x86-64 | WPE 后端仅 Linux（`dlopen`） |
| 编译器 | GCC 15.2.0 | 需支持 C++17 |
| CMake | 4.2.3 | ≥ 3.28 即可 |
| CUDA Toolkit | 13.4（nvcc 13.4.92） | `GMX_GPU=CUDA` |
| GPU | 需可见 NVIDIA GPU | 构建可用 `CUDA_ARCH` 指定，无需运行时设备 |
| FFTW | 系统 FFTW3（双/单精度） | 构建脚本用系统 FFTW，避免下载 |
| WPE（可选） | 1.0.0（`wpe.so/lib`） | 私有二进制，运行时加载；建议与 CUDA 运行库匹配 |

## 2. 源码布局

```text
src/external/gpumd/          最新 GPUMD master（未修改的 vendored 源码树）
src/external/wpe/            WPE 公开 C 头文件（BSD-2）+ LICENSE/NOTICE（不含私有二进制）
src/gromacs/applied_forces/nnpot/
  nepmodel.{h,cu}            NEP/qNEP 后端（GPUMD + WPE 统一入口，懒创建）
  nepmodelinfo.{h,cpp}       CPU 侧模型解析/校验（旧格式 + 新格式）
  nepmodel_qnep_gromacs.*    qNEP-GROMACS 链式力适配器（GPUMD 后端）
  wpeloader.{h,cpp}          WPE dlopen 加载器（版本/ABI 检查、错误信息）
  wpepotential.{h,cu}        WPE 的 GPUMD Potential 适配（零拷贝设备缓冲）
  nepbackendsettings.{h,cpp} 后端选择与选项结构
```

## 3. 构建命令

```bash
cd gromacs-2026.4-nep

# 默认：Release + WPE 后端 + 自动 -march=native
CUDA_ARCH=89 ./build-nep.sh

# 不编译 WPE（仍可用 GPU NEP/qNEP）
WPE=0 CUDA_ARCH=89 ./build-nep.sh

# 指定 WPE 运行库目录（仅写入 CMake 提示，不链接）
CUDA_ARCH=89 WPE_LIBRARY=/opt/wpe.so/lib ./build-nep.sh

# 调试构建
BUILD_TYPE=Debug CUDA_ARCH=89 ./build-nep.sh

# 严格数值（默认已关闭 fast math；如需更快可 FAST_MATH=1，但会改变数值）
CUDA_ARCH=89 FAST_MATH=0 ./build-nep.sh

# 安装
DO_INSTALL=1 CUDA_ARCH=89 INSTALL_PREFIX=/opt/gmx-nep ./build-nep.sh
```

等价的手工 CMake：

```bash
cmake -S . -B build-nep \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGMX_GPU=CUDA -DGMX_NNPOT=NEP -DGMX_NEP_WPE=ON \
  -DGMX_MPI=OFF -DGMX_THREAD_MPI=ON -DGMX_OPENMP=ON \
  -DGMX_FFT_LIBRARY=FFTW3 -DGMX_SIMD=AVX2_256 \
  -DBUILD_TESTING=ON -DGMX_BUILD_UNITTESTS=ON \
  -DREGRESSIONTEST_DOWNLOAD=OFF
cmake --build build-nep --parallel
```

## 4. 运行库配置

- GROMACS 自身的 `libgromacs.so`：若直接运行构建树中的 `gmx`，一般无需设置；
  若安装后库不在系统路径：
  ```bash
  export LD_LIBRARY_PATH=/opt/gmx-nep/lib:$LD_LIBRARY_PATH
  ```
- WPE：`gmx` **不链接** `libwisevolve_potential.so`，运行时按以下顺序查找：
  1. `nnpot-wpe-library`（mdp，可给目录或 `.so` 文件）
  2. `$GMX_WPE_LIBRARY`
  3. `$GMX_WPE_LIBRARY_PATH` / `$LD_LIBRARY_PATH` 中的 `libwisevolve_potential.so`
  4. 常见系统目录、`gmx` 可执行文件同级的 `../lib`
  5. 动态链接器默认搜索（`ldconfig`）
- 检查链接结果：
  ```bash
  ldd build-nep/bin/gmx | grep -E 'wisevolve|cuda|cufft'
  # 预期：只有 cudart/cufft；wisevolve 由运行时 dlopen，不出现
  ```

## 4.1 更新 vendored GPUMD

```bash
rm -rf src/external/gpumd && cp -a /path/to/GPUMD-master src/external/gpumd
CUDA_ARCH=89 ./build-nep.sh
```

详见 `docs/GPUMD-UPGRADE.md`。grompp 不需要 GPU（见 README 4.2 节）。

## 5. 验证

```bash
ctest --test-dir build-nep -R NNPotAppliedForcesUnitTest --output-on-failure
GMX=$PWD/build-nep/bin/gmx WPE_LIB=/opt/wpe.so/lib test/2026.4-validation/run_cpu_validation.sh
python3 test/2026.4-validation/wpe_abi_probe.py /opt/wpe.so/lib/libwisevolve_potential.so
```

## 6. 常见错误

| 现象 | 原因 / 处理 |
|---|---|
| `Could not detect the NVIDIA GPU compute capability` | 无 `nvidia-smi`；显式设置 `CUDA_ARCH`（如 89 / 120） |
| `The WPE backend was requested but libwisevolve_potential.so was not found` | 设置 `nnpot-wpe-library` 或 `GMX_WPE_LIBRARY` |
| `failed to load ... libcufft.so.11: cannot open shared object file` | WPE 包与 CUDA 运行库不匹配（用于 CUDA 12，系统为 CUDA 13）；安装 CUDA 12 运行库或获取匹配的 WPE 构建 |
| `WPE requires l_max_3body = 4` / `ANN hidden width <= 256` | 模型超出 WPE 形状约束；改用 `nnpot-backend = gpumd` |
| `nnpot-gromacs-charges needs the vendored GPUMD backend` | WPE 在引擎内部算 qNEP 静电；改用 `nnpot-backend = gpumd` 或原生 qNEP |
| `The NEP/qNEP backend needs a visible CUDA device in mdrun` | mdrun 需要 GPU；grompp/模型校验不需要 |
| `GPUMD removed NEP5 ...` | 最新 GPUMD 的 MD 只支持 NEP4；用 GPUMD 副本交换工具或重训 |
