# Newton SDF Inspector

该工具接收一个 STL 文件，调用 Newton 当前的 texture-SDF 构建流程，并将实际插值后的 SDF 场、Newton Marching Cubes 零等值面和三个中心剖面导出为 ParaView 可读取的 VTK XML 文件。

## 依赖

在 Newton 仓库根目录执行：

```bash
uv sync --extra importers
```

STL 读取依赖 `trimesh`，它包含在 `importers` extra 中。ParaView 可视化不依赖 VTK Python 或 PyVista。

### 编译原生 C++ 导出器（必须）

VTK 文件写入已从 Python 迁移到独立的 C++17 动态库。先在仓库根目录编译：

Linux：

```bash
cmake -S scripts/sdf_visualizer -B scripts/sdf_visualizer/build -DCMAKE_BUILD_TYPE=Release
cmake --build scripts/sdf_visualizer/build --parallel
ctest --test-dir scripts/sdf_visualizer/build --output-on-failure
```

Windows（Visual Studio 生成器）：

```powershell
cmake -S scripts/sdf_visualizer -B scripts/sdf_visualizer/build -A x64
cmake --build scripts/sdf_visualizer/build --config Release --parallel
ctest --test-dir scripts/sdf_visualizer/build -C Release --output-on-failure
```

默认情况下，Python 入口会自动查找 `scripts/sdf_visualizer/build` 中的动态库。也可以通过 `--native-writer` 或环境变量 `NEWTON_SDF_VTK_WRITER_LIBRARY` 指定库路径。

C++ 导出器直接从连续内存缓冲区写入 VTK XML 的 raw-appended binary 数据，不执行 Base64 编码，也不构造大型 Python XML/bytes 副本。写入采用临时文件后重命名，避免中断时留下半个最终文件。

CMake 会自动查找 CUDA Toolkit。若找到 CUDA Runtime，动态库会额外启用设备指针接口：Python 只传递 Warp 数组的 `ptr`、设备 ordinal 和当前 `wp.get_stream(device).cuda_stream`；C++ 在同一 CUDA stream 上执行分块 `cudaMemcpyAsync`，使用双缓冲 pinned host memory，并在磁盘写入当前块时传输下一块。每个 pinned buffer 上限为 32 MiB。若构建环境没有 CUDA Toolkit，则保留 C++ host-buffer 写入路径，Python 会回退到 Warp 的 host copy。

### 计算与 I/O 的职责边界

- SDF 构建继续调用 Newton 的现有 API。
- 三维场和切片采样继续运行在 Warp CUDA kernel 上，直接调用 Newton 的 `texture_sample_sdf`，不会在 CPU 上重新计算距离场。
- VTP/VTI/VTM 序列化由 C++17 动态库完成。
- GPU 采样结果仍需传输到主机内存后写入普通文件；这次迁移消除了 Python Base64/XML 序列化开销，但不会消除必要的 GPU→CPU 数据传输。

## 快速开始

```bash
python scripts/visualize_sdf.py \
  --input cases/hollow_cylinder.stl \
  --output outputs/hollow_cylinder \
  --input-unit m \
  --resolution 256 \
  --margin 0.02 \
  --narrow-band 0.05 \
  --sign-method auto
```

也可以使用 `--case` 作为 `--input` 的别名。

### 双场 SDF 可视化

传入第二个 STL 可分别构建 A、B 两个 Newton texture-SDF，并在两个纹理有效 AABB 的重叠区域内独立插值后再做交集组合。若两个有效区域没有重叠，程序会停止并提示增大 `--margin`（必要时同时调整 `--narrow-band`），而不是使用纹理域外的外推值生成误导性的 Combined 场：

```bash
uv run python scripts/visualize_sdf.py \
  --input cases/boundary_a.stl \
  --input-b cases/boundary_b.stl \
  --output outputs/dual_field \
  --input-unit mm \
  --target-voxel-size 0.0005
```

`--input-b` 与 `--input` 必须处于相同世界坐标系、使用相同单位和方向约定；工具不会自动配准两个网格。两个场均采用负值表示内部时，组合定义为：

`d_combined(x) = max(d_A(x), d_B(x))`

关键点是先分别插值 `d_A` 和 `d_B`，再取最大值；不要先在纹理顶点处做 max 再插值，否则同一个网格单元内的两个零交叉可能被抹掉。max 组合保留了交集的符号与零等值面，但在非光滑交线附近不保证仍是精确的欧氏距离场。

双场模式输出 `sdf_field_a.vti`、`sdf_field_b.vti`、`sdf_combined.vti`，以及三个中心切片对应的 A/B/Combined 文件。Newton 分别提取 A、B 的零等值面；Combined 零面可在 ParaView 中对 `sdf_combined.vti` 使用 Contour，等值设为 0。具体采样边界、组合方式和各场统计写入 `metadata.json`。

STL 不包含物理单位信息。若 STL 坐标以毫米为单位，必须指定 `--input-unit mm`；工具会先将坐标转换为米，再交给 Newton 构建 SDF。

若需要直接控制物理体素尺寸，可以使用：

```bash
python scripts/visualize_sdf.py \
  --input cases/hollow_cylinder.stl \
  --output outputs/hollow_cylinder_2mm \
  --input-unit m \
  --target-voxel-size 0.002
```

`--target-voxel-size` 与 `--resolution` 互斥，单位为米。长细结构应优先考虑物理体素尺寸，而不是只看最长轴分辨率。

对于壁厚约 2.5 mm 的薄壁结构，可以先尝试 0.5 mm 的目标体素尺寸（壁厚方向约有 5 个采样间隔），并将窄带控制在 5 mm 左右：

```bash
python scripts/visualize_sdf.py \
  --input cases/hollow_cylinder.stl \
  --output outputs/hollow_cylinder_0p5mm \
  --input-unit m \
  --target-voxel-size 0.0005 \
  --margin 0.01 \
  --narrow-band 0.005 \
  --slice-resolution 768
```

这是用于诊断的起始参数，不是对所有 STL 的最佳配置。高分辨率会增加 SDF 构建时间和显存占用；建议先用较低分辨率确认流程，再逐步减小体素尺寸。

## 输出文件

```text
outputs/hollow_cylinder/
├── case.vtm
├── source_mesh.vtp
├── newton_zero_isosurface.vtp
├── sdf_volume.vti
├── slices/
│   ├── sdf_slice_xy.vti
│   ├── sdf_slice_xz.vti
│   └── sdf_slice_yz.vti
├── metadata.json
└── README.txt
```

- `source_mesh.vtp`：由 STL 读取的源三角网格。
- `newton_zero_isosurface.vtp`：调用 Newton 的 `SDF.extract_isomesh(0)` 得到的零等值面。
- `sdf_volume.vti`：单场模式下，在规则三维采样点上调用 Newton texture-SDF 插值函数得到的标量场。
- 双场模式额外生成 `sdf_field_a.vti`、`sdf_field_b.vti` 和 `sdf_combined.vti`，分别表示独立采样的 A、B 与 max(A, B)。
- `sdf_slice_*.vti`：单场模式的 AABB 中心剖面；双场模式会为 A、B、Combined 分别输出 XY、XZ、YZ 剖面。
- `case.vtm`：ParaView 多数据集入口，打开一次即可加载所有数据。
- `metadata.json`：输入网格拓扑信息、单位、SDF 参数、纹理尺寸、采样尺寸、距离范围和耗时。

## ParaView 检查步骤

1. 打开 `case.vtm`。
2. 对比 `Source STL mesh` 与 `Newton extracted zero isosurface`，检查零面是否缺失、断裂或偏移。
3. 选中 `sdf_volume.vti`，应用 Contour Filter，将等值设为 `0.0`，得到基于导出标量场的零面。
4. 分别打开三个中心剖面，按 `signed_distance_m` 着色。对于闭合实体，材料内部应为负值，外部应为正值。
5. 查看 `metadata.json`，确认 STL 单位、watertight 状态、体素尺寸、Subgrid 数量及采样范围。

## 分辨率说明

`--resolution` 控制 Newton 实际构建 SDF 时最长 AABB 轴的最大分辨率；`--volume-resolution` 和 `--slice-resolution` 只控制导出可视化时的采样密度，不会提高已经构建好的 SDF 精度。

完整三维体数据受 `--max-volume-points` 限制；三个剖面分别受 `--max-slice-points` 限制。对于长细圆柱，中心 XY 剖面能以较高二维分辨率观察横截面，而无需生成一个极其庞大的高分辨率三维体数据。

当前工具可视化的是 Newton 已有的单场 texture-SDF 路径。实验性 Dual-Field 采样函数尚未接入完整的双纹理构建流程，因此本工具暂不把单场结果标记为 Dual-Field 结果。
