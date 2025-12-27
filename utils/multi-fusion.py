import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# 获取脚本所在目录，确保相对路径正确
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# 1. 读取 RGB & Depth
# =====================================================
rgb_path = os.path.join(SCRIPT_DIR, "rgb1.png")
rgb = cv2.imread(rgb_path)
assert rgb is not None, f"rgb.png not found at {rgb_path}"

rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

depth_path = os.path.join(SCRIPT_DIR, "depth1.png")
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
assert depth_raw is not None, f"depth.png not found at {depth_path}"

# 保存原始彩色深度图用于可视化
if len(depth_raw.shape) == 3:
    depth_color_original = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # 转为灰度用于计算
    depth = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
else:
    depth = depth_raw.astype(np.float32)
    # 如果原本是灰度，生成彩色版本
    depth_norm_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_color_original = plt.cm.jet(depth_norm_vis)[..., :3].astype(np.float32)

# =====================================================
# 统一所有图像尺寸（以 RGB 为基准）
# =====================================================
H_rgb, W_rgb = rgb.shape[:2]
H_depth, W_depth = depth.shape[:2]

if (H_rgb, W_rgb) != (H_depth, W_depth):
    print(f"⚠️ 尺寸不匹配: RGB={rgb.shape[:2]}, Depth={depth.shape[:2]}")
    print(f"   统一调整到 RGB 尺寸: ({H_rgb}, {W_rgb})")
    # 调整 depth 和 depth_color_original 到 RGB 尺寸
    depth = cv2.resize(depth, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
    depth_color_original = cv2.resize(depth_color_original, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)

# 归一化深度到 0–10 米
depth = depth - depth.min()
depth = depth / (depth.max() + 1e-6)
depth = depth * 10.0

H, W = depth.shape
print(f"✓ 统一尺寸: H={H}, W={W}")

# =====================================================
# 2. 生成空间不均匀烟雾 Mask
# =====================================================
def generate_smoke_mask(h, w, sigma=80):
    noise = np.random.rand(h, w)
    smoke = gaussian_filter(noise, sigma=sigma)
    smoke = (smoke - smoke.min()) / (smoke.max() - smoke.min() + 1e-6)
    return smoke

smoke_mask = generate_smoke_mask(H, W)

# =====================================================
# 3. 火焰检测（提前进行，用于烟雾渲染）
# =====================================================
def detect_fire_for_smoke(rgb_img):
    """
    检测火焰区域，用于在烟雾渲染时保护火焰
    """
    r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
    brightness = (r + g + b) / 3
    
    # 火焰核心（白/黄色，非常亮）
    fire_core = (brightness > 0.7) & (r > 0.6) & (g > 0.4)
    
    # 火焰边缘（橙/红色）
    fire_edge = (r > 0.5) & (r > g * 1.2) & (g > b * 1.1) & (brightness > 0.3)
    
    # 深红/暗火焰
    fire_dark = (r > 0.4) & (r > g * 1.5) & (r > b * 2.0)
    
    # 合并火焰区域
    fire_region = (fire_core | fire_edge | fire_dark).astype(np.float32)
    
    # 稍微扩展火焰区域，确保边缘也被保护
    fire_region = gaussian_filter(fire_region, sigma=5)
    fire_region = np.clip(fire_region * 1.5, 0, 1)
    
    return fire_region

# 预先检测火焰区域
fire_protection_mask = detect_fire_for_smoke(rgb)

# =====================================================
# 4. 烟雾 RGB 合成（火焰区域保护）
# =====================================================
def apply_smoke(rgb, depth, smoke_mask, fire_protection, beta=0.6):
    """
    Koschmieder-inspired smoke model
    烟雾会遮挡普通物体，但不会遮挡火焰（火焰光亮穿透烟雾）
    """
    A = np.array([0.75, 0.75, 0.75])  # 空气光（灰白）
    transmission = np.exp(-beta * depth)
    transmission = transmission * (0.3 + 0.7 * smoke_mask)
    
    # 烟雾效果应用到非火焰区域
    out_smoke = rgb * transmission[..., None] + A * (1 - transmission[..., None])
    
    # 火焰区域：保持原始颜色，只添加轻微的烟雾散射光晕
    # 火焰的光会穿透烟雾，但会产生一些光晕效果
    fire_glow = gaussian_filter(fire_protection, sigma=15)  # 火焰光晕
    
    # 在火焰区域，烟雾效果大幅减弱
    # fire_protection: 1 = 火焰区域（几乎无烟雾效果），0 = 普通区域（正常烟雾）
    smoke_reduction = 1 - fire_protection * 0.9  # 火焰区域烟雾减少90%
    
    # 混合：火焰区域保留更多原始颜色
    out = out_smoke * smoke_reduction[..., None] + rgb * (1 - smoke_reduction[..., None])
    
    # 火焰光晕效果：在烟雾中火焰周围有暖色光晕
    glow_color = np.array([1.0, 0.6, 0.3])  # 暖橙色光晕
    glow_intensity = fire_glow * smoke_mask * 0.3  # 烟雾中的光晕
    out = out + glow_color * glow_intensity[..., None]
    
    return np.clip(out, 0, 1)

def apply_smoke_to_depth(depth_color, depth_values, smoke_mask, beta=0.4):
    """
    给彩色深度图应用烟雾效果
    烟雾会降低深度图的对比度和清晰度，模拟传感器受烟雾干扰
    """
    A = np.array([0.6, 0.6, 0.6])  # 烟雾中深度传感器的噪声颜色（灰色）
    
    # 基于深度的透射率
    depth_norm = depth_values / (depth_values.max() + 1e-6)
    transmission = np.exp(-beta * depth_norm * 5)
    
    # 烟雾密度影响
    transmission = transmission * (0.4 + 0.6 * (1 - smoke_mask))
    transmission = np.clip(transmission, 0.1, 1.0)
    
    # 应用烟雾效果
    out = depth_color * transmission[..., None] + A * (1 - transmission[..., None])
    
    # 添加一些噪声模拟传感器干扰
    noise = np.random.normal(0, 0.02, out.shape)
    out = out + noise * smoke_mask[..., None]
    
    return np.clip(out, 0, 1)

rgb_smoke = apply_smoke(rgb, depth, smoke_mask, fire_protection_mask)
depth_smoke = apply_smoke_to_depth(depth_color_original, depth, smoke_mask)

# =====================================================
# 5. Thermal（热成像）合成 - 增强版
# =====================================================
def detect_fire_from_rgb(rgb_img):
    """
    从 RGB 图像中检测火焰区域（基于暖色调）
    返回火焰强度 mask (0-1)
    """
    # rgb_img 是 0-1 范围的 float32
    r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
    
    # 火焰检测条件：
    # 1. 红色通道高
    # 2. 红色 > 绿色 > 蓝色（火焰的典型特征）
    # 3. 亮度较高
    
    # 火焰核心（白/黄色，非常亮）
    brightness = (r + g + b) / 3
    fire_core = (brightness > 0.7) & (r > 0.6) & (g > 0.4)
    
    # 火焰边缘（橙/红色）
    fire_edge = (r > 0.5) & (r > g * 1.2) & (g > b * 1.1) & (brightness > 0.3)
    
    # 深红/暗火焰
    fire_dark = (r > 0.4) & (r > g * 1.5) & (r > b * 2.0)
    
    # 合并火焰区域
    fire_mask = fire_core.astype(np.float32) * 1.0 + \
                fire_edge.astype(np.float32) * 0.7 + \
                fire_dark.astype(np.float32) * 0.5
    
    fire_mask = np.clip(fire_mask, 0, 1)
    
    # 平滑火焰边界
    fire_mask = gaussian_filter(fire_mask, sigma=3)
    fire_mask = (fire_mask - fire_mask.min()) / (fire_mask.max() - fire_mask.min() + 1e-6)
    
    return fire_mask

def extract_edge_from_rgb(rgb_img):
    """
    从 RGB 图像提取边缘轮廓
    """
    # 转为灰度
    gray = 0.299 * rgb_img[:,:,0] + 0.587 * rgb_img[:,:,1] + 0.114 * rgb_img[:,:,2]
    gray_uint8 = (gray * 255).astype(np.uint8)
    
    # Canny 边缘检测
    edges = cv2.Canny(gray_uint8, 30, 100)
    
    # 轻微膨胀使边缘更明显
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 归一化到 0-1
    edges = edges.astype(np.float32) / 255.0
    
    return edges

def generate_thermal_enhanced(rgb_img, depth, smoke_mask):
    """
    增强版热成像生成：
    - 从 RGB 检测火焰
    - 显示场景轮廓
    - 火焰区域温度更高更鲜明
    输出单位：Kelvin（模拟）
    """
    H, W = depth.shape
    
    # 1. 基础环境温度（根据深度，近处稍暖）
    depth_norm = depth / (depth.max() + 1e-6)
    thermal = 290 + 15 * (1 - depth_norm)  # 290-305K 环境温度
    
    # 2. 从 RGB 检测火焰
    fire_mask = detect_fire_from_rgb(rgb_img)
    
    # 3. 火焰温度叠加（火焰核心可达 1200K+）
    # 使用非线性映射使火焰更鲜明
    fire_intensity = fire_mask ** 0.5  # 增强弱火焰的可见度
    thermal += 900 * fire_intensity  # 火焰区域大幅升温
    
    # 4. 额外的火焰热点（模拟不均匀燃烧）
    if fire_mask.max() > 0.1:
        # 在检测到火焰的区域添加随机热点
        fire_hotspots = fire_mask * np.random.uniform(0.8, 1.2, size=(H, W))
        fire_hotspots = gaussian_filter(fire_hotspots, sigma=5)
        thermal += 200 * fire_hotspots
    
    # 5. 提取并叠加 RGB 轮廓（轻微温度变化显示结构）
    edges = extract_edge_from_rgb(rgb_img)
    # 边缘区域有轻微温差（模拟材质边界的热传导差异）
    thermal += 20 * edges
    
    # 6. 物体表面温度变化（基于 RGB 亮度）
    brightness = (rgb_img[:,:,0] + rgb_img[:,:,1] + rgb_img[:,:,2]) / 3
    # 亮色物体反射更多光，温度稍低；暗色物体吸热更多
    thermal += 10 * (0.5 - brightness)
    
    # 7. 烟雾对 thermal 的轻微衰减
    thermal -= 30 * smoke_mask
    
    # 8. 添加轻微噪声模拟真实热成像
    noise = np.random.normal(0, 2, thermal.shape)
    thermal += noise
    
    return thermal, fire_mask, edges

thermal, fire_mask, edges = generate_thermal_enhanced(rgb, depth, smoke_mask)

# =====================================================
# 6. 可视化
# =====================================================
plt.figure(figsize=(18, 8))

# 第一行：RGB 和 Depth 对比
plt.subplot(2, 4, 1)
plt.title("Original RGB")
plt.imshow(rgb)
plt.axis("off")

plt.subplot(2, 4, 2)
plt.title("RGB + Smoke")
plt.imshow(rgb_smoke)
plt.axis("off")

plt.subplot(2, 4, 3)
plt.title("Original Depth (Color)")
plt.imshow(depth_color_original)
plt.axis("off")

plt.subplot(2, 4, 4)
plt.title("Depth + Smoke")
plt.imshow(depth_smoke)
plt.axis("off")

# 第二行：烟雾和热成像
plt.subplot(2, 4, 5)
plt.title("Smoke Density")
plt.imshow(smoke_mask, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.subplot(2, 4, 6)
plt.title("Fire Detection")
plt.imshow(fire_mask, cmap="hot")
plt.colorbar()
plt.axis("off")

plt.subplot(2, 4, 7)
plt.title("Thermal (K)")
plt.imshow(thermal, cmap="inferno")
plt.colorbar()
plt.axis("off")

plt.subplot(2, 4, 8)
plt.title("Depth Values (m)")
plt.imshow(depth, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "multi_fusion_result.png"), dpi=150)
# plt.show()  # 取消注释以交互显示

# =====================================================
# 7. 保存结果
# =====================================================
output_dir = SCRIPT_DIR

# RGB 烟雾效果
cv2.imwrite(os.path.join(output_dir, "rgb_smoke.png"), 
            (rgb_smoke * 255).astype(np.uint8)[:, :, ::-1])

# 彩色深度图 + 烟雾效果
cv2.imwrite(os.path.join(output_dir, "depth_smoke.png"), 
            (depth_smoke * 255).astype(np.uint8)[:, :, ::-1])

# 原始彩色深度图（保留）
cv2.imwrite(os.path.join(output_dir, "depth_color_original.png"), 
            (depth_color_original * 255).astype(np.uint8)[:, :, ::-1])

# 热成像
thermal_norm = (thermal - thermal.min()) / (thermal.max() - thermal.min())
thermal_color = plt.cm.inferno(thermal_norm)[..., :3]
cv2.imwrite(os.path.join(output_dir, "thermal.png"), 
            (thermal_color * 255).astype(np.uint8)[:, :, ::-1])

np.save(os.path.join(output_dir, "thermal.npy"), thermal)

print("✅ 生成完成：")
print(" - rgb_smoke.png          : RGB + 烟雾效果")
print(" - depth_smoke.png        : 彩色深度图 + 烟雾效果")
print(" - depth_color_original.png : 原始彩色深度图")
print(" - thermal.png            : 热成像")
print(" - multi_fusion_result.png : 完整对比图")
