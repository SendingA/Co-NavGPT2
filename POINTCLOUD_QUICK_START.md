# 点云可视化快速启动指南

## 📊 核心概念

你的项目中点云的生成和流动过程：

```
深度图 (480×640)
    ↓
[Agent 1 mapping()] → point_sum_1 (彩色点云)
[Agent 2 mapping()] → point_sum_2 (彩色点云)
[Agent 3 mapping()] → point_sum_N (彩色点云)
    ↓
main.py 中汇总 → point_sum (所有 Agent 点云合并)
    ↓
可视化：GUI 或 导出 PLY 文件
```

---

## 🚀 快速开始

### 方式 1：实时 Open3D GUI（推荐）

```bash
# 启用可视化 GUI
python main.py --visualize --nav_mode gpt

# 特点：
# ✓ 实时看到点云
# ✓ 可以交互操作（旋转/缩放/平移）
# ✓ 显示多 Agent 相机轨迹
# ✗ 资源占用较多
```

**GUI 界面说明**：
- 左侧面板：RGB Point Cloud 复选框（勾选显示点云）
- 3D 视口：中心的点云 + 蓝色相机视锥 + 绿色轨迹
- 上方选项卡：切换"Annotated Image" / "Semantic Maps"

---

### 方式 2：导出 PLY 文件（推荐用于分析）

**集成到 main.py**（无需 GUI 开销）：

在 `main.py` 中找到：
```python
while not env.episode_over:
    # ... 现有代码 ...
    for i in range(num_agents):
        agent[i].mapping(observations[i], agent_state)
        point_sum += agent[i].point_sum
```

添加导出代码：
```python
# 在上述代码后添加：
if count_step % 10 == 0:  # 每 10 步保存一次
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(point_sum.points))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(point_sum.colors))
    
    output_dir = f"{args.dump_location}/pointclouds"
    os.makedirs(output_dir, exist_ok=True)
    o3d.io.write_point_cloud(
        f"{output_dir}/frame_{count_episodes:03d}_{count_step:06d}.ply", pcd
    )
```

**查看导出的点云**：
```bash
# 方式 A：用 Python 脚本查看
python -c "import open3d as o3d; \
           pcd = o3d.io.read_point_cloud('logs/gpt/pointclouds/frame_000_000000.ply'); \
           o3d.visualization.draw_geometries([pcd])"

# 方式 B：用 CloudCompare（图形化工具）
# 下载：https://www.cloudcompare.org/
# 拖拽 .ply 文件即可打开

# 方式 C：用 Meshlab
# 下载：https://www.meshlab.net/
```

---

### 方式 3：使用提供的可视化工具

```python
# 1. 导入
from utils.pointcloud_vis import PointCloudVisualizer

# 2. 初始化
vis = PointCloudVisualizer(output_dir="logs/pointclouds")

# 3. 导出原始点云
vis.export_point_cloud(points, colors, "my_pointcloud")

# 4. 按距离着色
colors_dist = vis.color_by_distance(points, reference_point=[0,0,0])
vis.export_point_cloud(points, colors_dist, "pointcloud_by_distance")

# 5. 按高度着色
colors_height = vis.color_by_height(points)
vis.export_point_cloud(points, colors_height, "pointcloud_by_height")

# 6. Open3D 显示
vis.visualize_with_open3d([{'points': points, 'colors': colors}])
```

---

## 📁 文件位置说明

| 文件 | 功能 | 关键部分 |
|------|------|---------|
| `agents/vlm_agents.py` | Agent 点云生成 | `mapping()`, `self.point_sum` |
| `utils/mapping.py` | 点云处理工具 | `create_object_pcd()` |
| `utils/explored_map_utils.py` | 全景点云构建 | `build_full_scene_pcd()` |
| `utils/pointcloud_vis.py` | **⭐ 可视化工具** | `PointCloudVisualizer` 类 |
| `utils/vis_gui.py` | Open3D GUI | `ReconstructionWindow` |
| `main.py` | 主循环 | 点云汇总逻辑 |

---

## 🔍 数据结构

### Open3D PointCloud
```python
import open3d as o3d
import numpy as np

pcd = o3d.geometry.PointCloud()

# 设置点坐标（N×3 浮点数组，单位：米）
pcd.points = o3d.utility.Vector3dVector(points)  # shape: (N, 3)

# 设置颜色（N×3 浮点数组，范围：[0, 1]）
pcd.colors = o3d.utility.Vector3dVector(colors)  # shape: (N, 3)

# 转换为 numpy 数组操作
points_np = np.asarray(pcd.points)    # (N, 3) float64
colors_np = np.asarray(pcd.colors)    # (N, 3) float64

# 基本操作
pcd.voxel_down_sample(voxel_size=0.05)  # 体素下采样
pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)  # 去离群值
pcd.estimate_normals()  # 估计法向量
```

---

## 💡 实用技巧

### 1. 点云太多/太慢？使用下采样
```python
# 每 0.05m 保留一个点
pcd.voxel_down_sample(voxel_size=0.05)

# 随机下采样
downsampled = pcd.random_down_sample(sampling_ratio=0.1)
```

### 2. 多 Agent 点云对比着色
```python
from utils.pointcloud_vis import PointCloudVisualizer

vis = PointCloudVisualizer()

# 预定义的 Agent 颜色
agent_pcd_dict = {
    0: {'points': agent_0_points, 'colors': np.full(agent_0_points.shape, [1, 0, 0])},  # 红
    1: {'points': agent_1_points, 'colors': np.full(agent_1_points.shape, [0, 1, 0])},  # 绿
    2: {'points': agent_2_points, 'colors': np.full(agent_2_points.shape, [0, 0, 1])},  # 蓝
}

merged_points, merged_colors = vis.merge_agent_point_clouds(agent_pcd_dict)
vis.export_point_cloud(merged_points, merged_colors, "multi_agent_comparison")
```

### 3. 热力图着色（按距离或高度）
```python
# 按摄像机距离着色（热力）
colors = vis.color_by_distance(
    points, 
    reference_point=camera_position,
    colormap='hot'  # 'viridis', 'plasma', 'inferno', 'magma' 等
)

# 按高度着色（梯度颜色）
colors = vis.color_by_height(points, colormap='viridis')
```

### 4. 过滤太远的点
```python
# 只保留距离相机 5m 以内的点
filtered_points, filtered_colors = vis.filter_points_by_distance(
    points, colors,
    max_distance=5.0,
    reference_point=camera_position
)
```

### 5. 保存为其他格式
```python
# PLY（推荐，保留颜色信息）
o3d.io.write_point_cloud("cloud.ply", pcd)

# PCD（ROS 常用格式）
o3d.io.write_point_cloud("cloud.pcd", pcd)

# XYZ（简单文本格式）
o3d.io.write_point_cloud("cloud.xyz", pcd)

# 读取
pcd = o3d.io.read_point_cloud("cloud.ply")
```

---

## 🐛 常见问题排查

### 问题 1：点云看不见（全黑）
```python
# 检查点云是否为空
if len(pcd.points) == 0:
    print("ERROR: 没有点!")
else:
    print(f"OK: {len(pcd.points)} 个点")

# 检查颜色范围
colors = np.asarray(pcd.colors)
print(f"颜色范围: [{colors.min()}, {colors.max()}]")
# 应该在 [0, 1] 之间
```

### 问题 2：点数特别多导致卡顿
```python
# 下采样
pcd.voxel_down_sample(0.1)  # 10cm 体素

# 或随机采样
downsampled = pcd.random_down_sample(0.05)  # 5% 采样率
```

### 问题 3：点云坐标异常（全在一个地方）
```python
# 检查坐标范围
points = np.asarray(pcd.points)
print(f"X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
print(f"Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
print(f"Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

# 检查是否正确进行了坐标变换
pcd.transform(camera_matrix_T)  # 4×4 位姿矩阵
```

### 问题 4：GUI 显示很卡
```bash
# 方案 A：禁用 GUI，改用文件导出
python main.py --no-visualize  # 改为存文件

# 方案 B：降低导出频率
# 改 main.py 中的 export_interval = 100（每 100 步导出一次）

# 方案 C：下采样
pcd.voxel_down_sample(0.1)
```

---

## 📊 性能参考

| 操作 | 点数 | 时间 | 内存 |
|------|------|------|------|
| 深度图→点云 | 307K | ~10ms | ~15MB |
| 下采样到 1cm | 307K→10K | ~5ms | ~0.5MB |
| 写 PLY 文件 | 10K | ~50ms | 磁盘 |
| Open3D 显示 | 10K | <1ms | 显存 ~100MB |

---

## 🎯 推荐工作流

1. **开发阶段**：
   ```bash
   # 关闭 GUI（加快运行）
   python main.py --nav_mode gpt
   # 导出 PLY 文件分析
   ```

2. **可视化阶段**：
   ```bash
   # 用 CloudCompare 打开 PLY 文件
   # 逐帧检查点云质量
   ```

3. **调试阶段**：
   ```bash
   # 启用 GUI，实时观察
   python main.py --visualize --nav_mode gpt
   ```

4. **最终报告**：
   ```bash
   # 导出高质量点云渲染图
   # 用 CloudCompare 的截图功能
   ```

---

## 📞 API 速查

```python
from utils.pointcloud_vis import PointCloudVisualizer

vis = PointCloudVisualizer(output_dir="logs/pointclouds")

# 导出
vis.export_point_cloud(points, colors, filename)

# 着色
colors_dist = vis.color_by_distance(points, ref_point)
colors_height = vis.color_by_height(points)

# 过滤
pts, cols = vis.filter_points_by_distance(points, colors, max_dist)

# 合并
merged_pts, merged_cols = vis.merge_agent_point_clouds(dict)

# 显示
vis.visualize_with_open3d([pcd_list])
vis.visualize_with_matplotlib(points, colors)
```

---

## 📖 相关文档

- **完整分析**: `POINTCLOUD_VISUALIZATION_GUIDE.md`
- **集成示例**: `examples/pointcloud_integration_example.py`
- **工具源码**: `utils/pointcloud_vis.py`
