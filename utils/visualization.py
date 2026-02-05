import math
from typing import Iterable
import dataclasses
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import glob

import numpy as np
from typing import List, Union
import skimage.morphology
from PIL import Image
from constants import color_palette, coco_categories, category_to_id

import supervision as sv
from supervision.draw.color import Color, ColorPalette


def images_to_video(image_dir, output_path, fps=10, pattern="Merged_Vis-*.png"):
    """
    将指定目录下的图片序列合成视频
    
    Args:
        image_dir: 图片所在目录
        output_path: 输出视频路径
        fps: 视频帧率
        pattern: 图片文件名匹配模式
    
    Returns:
        bool: 是否成功生成视频
    """
    # 获取所有图片文件
    image_files = glob.glob(os.path.join(image_dir, pattern))
    
    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到匹配 {pattern} 的图片")
        return False
    
    # 按数字排序 (提取文件名中的数字)
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        # 提取 "Merged_Vis-123.png" 或 "agent-0-Vis-123.png" 中的最后一个数字
        parts = filename.replace('.png', '').split('-')
        try:
            return int(parts[-1])
        except ValueError:
            return 0
    
    image_files.sort(key=extract_number)
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误: 无法读取图片 {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件 {output_path}")
        return False
    
    # 写入所有帧
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    print(f"视频已保存: {output_path} ({len(image_files)} 帧, {fps} FPS)")
    return True


def create_episode_video(args, episode_n, rank=0):
    """
    为指定 episode 创建视频:
    1. agent-0-Vis-*.png -> agent_0 视频
    2. agent-1-Vis-*.png -> agent_1 视频 (如果存在)
    3. Merged_Vis-*.png -> merged 视频
    4. agent-X-RGBD-*.png -> agent_X RGBD 视频
    
    Args:
        args: 命令行参数
        episode_n: episode 编号
        rank: 进程编号
    
    Returns:
        list: 成功生成的视频文件路径列表
    """
    dump_dir = "{}/dump/{}".format(args.dump_location, args.nav_mode)
    ep_dir = '{}/episodes_multi/{}/eps_{}/'.format(dump_dir, rank, episode_n)
    video_dir = '{}/videos/'.format(dump_dir)
    
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    fps = getattr(args, 'video_fps', 10)
    num_agents = getattr(args, 'num_agents', 2)
    
    video_paths = []
    
    # 1. 为每个 agent 生成普通可视化视频
    for agent_id in range(num_agents):
        pattern = f"agent-{agent_id}-Vis-*.png"
        video_path = os.path.join(video_dir, f'episode_{episode_n}_rank_{rank}_agent_{agent_id}.mp4')
        if images_to_video(ep_dir, video_path, fps=fps, pattern=pattern):
            video_paths.append(video_path)
    
    # 2. 为每个 agent 生成 RGBD 视频
    for agent_id in range(num_agents):
        pattern = f"agent-{agent_id}-RGBD-*.png"
        video_path = os.path.join(video_dir, f'episode_{episode_n}_rank_{rank}_agent_{agent_id}_rgbd.mp4')
        if images_to_video(ep_dir, video_path, fps=fps, pattern=pattern):
            video_paths.append(video_path)
    
    # 3. 生成 Merged 视频
    pattern = "Merged_Vis-*.png"
    video_path = os.path.join(video_dir, f'episode_{episode_n}_rank_{rank}_merged.mp4')
    if images_to_video(ep_dir, video_path, fps=fps, pattern=pattern):
        video_paths.append(video_path)
    
    # 4. 生成 TopView + Frontiers 视频
    pattern = "TopView_Frontiers-*.png"
    video_path = os.path.join(video_dir, f'episode_{episode_n}_rank_{rank}_topview_frontiers.mp4')
    if images_to_video(ep_dir, video_path, fps=fps, pattern=pattern):
        video_paths.append(video_path)
    
    return video_paths if video_paths else None


# Copied from https://github.com/concept-graphs/concept-graphs/     
def vis_result_fast(
    image: np.ndarray, 
    detections: sv.Detections, 
    classes: List[str], 
    color: Union[Color, ColorPalette] = ColorPalette.default(),
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    '''
    Annotate the image with the detection results. 
    This is fast but of the same resolution of the input image, thus can be blurry. 
    '''
    # Annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator(
        color=color,
        thickness=1  # Thickness of bounding box lines
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(
        color=color
    )
    
    # Generate labels
    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _
        in detections
    ]
    
    if instance_random_color:
        # Generate random colors for each instance
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
        
    # Apply mask annotations
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    # Apply bounding box annotations
    if draw_bbox:
        annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Apply text labels separately
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image

def init_vis_image(goal_name, action = 0):
    vis_image = np.ones((537, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations" 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Find {}  Action {}".format(goal_name, str(action))
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color


#     # draw legend
#     lx, ly, _ = legend.shape
#     vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image

def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat

def init_multi_vis_image(goal_name, multi_color, s_x = 537, s_y = 670):
    vis_image = np.ones((s_x, s_y, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Find {}".format(goal_name) 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 50
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    for i in range(len(multi_color)):
        text = "Agent {}".format(i) 
        vis_image = cv2.putText(vis_image, text, (textX+200+150*i, textY),
                                font, fontScale, multi_color[i], thickness,
                                cv2.LINE_AA)
    # draw outlines
    color = [100, 100, 100]
    # vis_image[49, 15:495] = color
    # vis_image[50:530, 14] = color
    # vis_image[50:530, 495] = color
    # vis_image[530, 15:495] = color


#     # draw legend
#     lx, ly, _ = legend.shape
#     vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])

EPS = 1e-4
def write_number(image, pose, number):
    
    pil_image = Image.fromarray(image)
        
    # add the number on the image
    # Initialize drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # 1. Draw the main number as a rectangle.
    font_size_main = 30
    try:
        font_main = ImageFont.truetype("arial.ttf", font_size_main)
    except IOError:
        font_main = ImageFont.load_default(font_size_main)

    text_width = 20
    text_height = 35
    padding = 3
    position = (3, 3)  # Adjust position as needed

    # Define the rectangle coordinates
    rect_x0 = position[0] - padding
    rect_y0 = position[1] - padding
    rect_x1 = position[0] + text_width + padding
    rect_y1 = position[1] + text_height + padding

    # Draw the white rectangle
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="white")

    # Add text to image
    draw.text(position, str(number), fill="red", font=font_main)

    # 2. Draw circles for each pose point.
    circle_radius = 12
    try:
        font_pose = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font_pose = ImageFont.load_default(15)

    drawn_centers = []
    def push_away(px, py, existing_x, existing_y, radius):
        """Push point (px, py) away from (existing_x, existing_y) just enough to not overlap."""
        dist = math.dist((px, py), (existing_x, existing_y))
        # If already not overlapping or same point, do nothing
        if dist >= 2 * radius:
            return px, py

        # Calculate overlap distance
        overlap = 2 * radius - dist
        # Direction from existing circle to new circle
        dx = px - existing_x
        dy = py - existing_y
        # If dx,dy is zero, pick a random small direction
        if dx == 0 and dy == 0:
            dx, dy = 1e-3, 0
        length = math.hypot(dx, dy)

        # Normalize direction, move 'overlap/2' away 
        # (or some fraction, depending how you want them spaced)
        nx = dx / length
        ny = dy / length
        px += nx * (overlap / 2)
        py += ny * (overlap / 2)
        return px, py
    
    for i, (px, py, pz) in enumerate(pose):
        # py = 480-py
        moved = True
        while moved:
            moved = False
            for (ex, ey) in drawn_centers:
                dist = math.dist((px, py), (ex, ey))
                if dist + EPS < 2 * circle_radius:
                    # push away
                    px, py = push_away(px, py, ex, ey, circle_radius)
                    moved = True
                
                
        # Circle bounding box
        x0 = px - circle_radius
        y0 = py - circle_radius
        x1 = px + circle_radius
        y1 = py + circle_radius

        # Draw the black-filled circle with a white outline
        draw.ellipse(
            [x0, y0, x1, y1],
            fill="white",
            outline="black",  # optional outline color
            width=2           # outline thickness
        )

        # Text in the center
        index_str = "R"+str(i)
        # Use textbbox or font.getsize
        bbox_pose = draw.textbbox((0, 0), index_str, font=font_pose)
        text_width_pose = bbox_pose[2] - bbox_pose[0]
        text_height_pose = bbox_pose[3] - bbox_pose[1]

        text_x_pose = px - text_width_pose / 2
        text_y_pose = py - text_height_pose +2

        draw.text((text_x_pose, text_y_pose), index_str, fill="black", font=font_pose)

        drawn_centers.append((px, py))
    
    return pil_image

def write_number_full(image, pose, number):
    
    pil_image = Image.fromarray(image)
        
    # add the number on the image
    # Initialize drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # 1. Draw the main number as a rectangle.
    font_size_main = 30
    try:
        font_main = ImageFont.truetype("arial.ttf", font_size_main)
    except IOError:
        font_main = ImageFont.load_default(font_size_main)

    text_width = 20
    text_height = 35
    padding = 3
    position = (3, 3)  # Adjust position as needed

    # Define the rectangle coordinates
    rect_x0 = position[0] - padding
    rect_y0 = position[1] - padding
    rect_x1 = position[0] + text_width + padding
    rect_y1 = position[1] + text_height + padding

    # Draw the white rectangle
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="white")

    # Add text to image
    draw.text(position, str(number), fill="red", font=font_main)

    # 2. Draw circles for each pose point.
    circle_radius = 12
    try:
        font_pose = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font_pose = ImageFont.load_default(15)

    drawn_centers = []
    def push_away(px, py, existing_x, existing_y, radius):
        """Push point (px, py) away from (existing_x, existing_y) just enough to not overlap."""
        dist = math.dist((px, py), (existing_x, existing_y))
        # If already not overlapping or same point, do nothing
        if dist >= 2 * radius:
            return px, py

        # Calculate overlap distance
        overlap = 2 * radius - dist
        # Direction from existing circle to new circle
        dx = px - existing_x
        dy = py - existing_y
        # If dx,dy is zero, pick a random small direction
        if dx == 0 and dy == 0:
            dx, dy = 1e-3, 0
        length = math.hypot(dx, dy)

        # Normalize direction, move 'overlap/2' away 
        # (or some fraction, depending how you want them spaced)
        nx = dx / length
        ny = dy / length
        px += nx * (overlap / 2)
        py += ny * (overlap / 2)
        return px, py
    
    for i, (px, py, pz) in enumerate(pose):
        # py = 480-py
        moved = True
        while moved:
            moved = False
            for (ex, ey) in drawn_centers:
                dist = math.dist((px, py), (ex, ey))
                if dist + EPS < 2 * circle_radius:
                    # push away
                    px, py = push_away(px, py, ex, ey, circle_radius)
                    moved = True
                
                
        # Circle bounding box
        x0 = px - circle_radius
        y0 = py - circle_radius
        x1 = px + circle_radius
        y1 = py + circle_radius

        # Draw the black-filled circle with a white outline
        draw.ellipse(
            [x0, y0, x1, y1],
            fill="white",
            outline="black",  # optional outline color
            width=2           # outline thickness
        )

        # Text in the center
        index_str = "R"+str(i)
        # Use textbbox or font.getsize
        bbox_pose = draw.textbbox((0, 0), index_str, font=font_pose)
        text_width_pose = bbox_pose[2] - bbox_pose[0]
        text_height_pose = bbox_pose[3] - bbox_pose[1]

        text_x_pose = px - text_width_pose / 2
        text_y_pose = py - text_height_pose +2

        draw.text((text_x_pose, text_y_pose), index_str, fill="black", font=font_pose)

        drawn_centers.append((px, py))
    
    return pil_image


def visualize_topview_with_frontiers(args, step, top_view_map, pose_pred, target_point_list, 
                                      goal_points=None, episode_n=0, rank=0):
    """
    在 top-view 图上优雅地标记 robots 位置和 candidate frontiers。
    
    Args:
        args: 命令行参数
        step: 当前步数
        top_view_map: 俯视图 (H, W, 3)，BGR 格式
        pose_pred: robots 的位置列表 [(x, y, theta), ...]
        target_point_list: candidate frontiers 列表 [(x, y), ...]
        goal_points: 当前目标点列表（可选）
        episode_n: episode 编号
        rank: 进程编号
    
    Returns:
        np.ndarray: 标记后的图像
    """
    # 复制图像避免修改原图
    vis_map = top_view_map.copy()
    
    # 调整到合适的显示尺寸
    display_size = 600
    h, w = vis_map.shape[:2]
    scale = display_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    vis_map = cv2.resize(vis_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 计算坐标缩放比例
    scale_x = new_w / w
    scale_y = new_h / h
    
    # 翻转图像（与其他可视化保持一致）
    vis_map = np.flipud(vis_map).copy()
    
    # ===== 1. 绘制 Candidate Frontiers =====
    if target_point_list and len(target_point_list) > 0:
        num_frontiers = len(target_point_list)
        
        for idx, frontier in enumerate(target_point_list):
            # 转换坐标
            fx, fy = frontier[0], frontier[1]
            fx_scaled = int(fx * scale_x)
            fy_scaled = new_h - int(fy * scale_y)  # 翻转 y
            
            # 检查是否是当前目标点
            is_goal = False
            if goal_points:
                for gp in goal_points:
                    if abs(gp[0] - frontier[0]) < 3 and abs(gp[1] - frontier[1]) < 3:
                        is_goal = True
                        break
            
            if is_goal:
                # 目标点：用星形或更大的标记
                color = (0, 255, 255)  # 黄色 (BGR)
                # 绘制星形
                star_size = 12
                pts = []
                for i in range(5):
                    # 外点
                    angle_out = -np.pi/2 + i * 2 * np.pi / 5
                    pts.append([fx_scaled + int(star_size * np.cos(angle_out)),
                               fy_scaled + int(star_size * np.sin(angle_out))])
                    # 内点
                    angle_in = angle_out + np.pi / 5
                    pts.append([fx_scaled + int(star_size * 0.4 * np.cos(angle_in)),
                               fy_scaled + int(star_size * 0.4 * np.sin(angle_in))])
                pts = np.array(pts, np.int32)
                cv2.fillPoly(vis_map, [pts], color)
                cv2.polylines(vis_map, [pts], True, (0, 200, 200), 2)
            else:
                # 普通 frontier：用小圆点
                color = (255, 200, 100)  # 浅蓝色 (BGR)
                cv2.circle(vis_map, (fx_scaled, fy_scaled), 6, color, -1)
                cv2.circle(vis_map, (fx_scaled, fy_scaled), 6, (200, 150, 50), 1)
            
            # 添加 frontier 编号（小字体）
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"F{idx}"
            font_scale = 0.35
            cv2.putText(vis_map, text, (fx_scaled + 8, fy_scaled + 3), 
                       font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # ===== 2. 绘制 Robots =====
    num_agents = len(pose_pred)
    robot_colors = [
        (0, 0, 255),    # 红色 - Robot 0
        (0, 255, 0),    # 绿色 - Robot 1
        (255, 0, 0),    # 蓝色 - Robot 2
        (255, 0, 255),  # 紫色 - Robot 3
    ]
    
    for i, pose in enumerate(pose_pred):
        px, py, theta = pose[0], pose[1], pose[2] if len(pose) > 2 else 0
        px_scaled = int(px * scale_x)
        py_scaled = new_h - int(py * scale_y)  # 翻转 y
        
        color = robot_colors[i % len(robot_colors)]
        
        # 绘制机器人主体（圆形）
        cv2.circle(vis_map, (px_scaled, py_scaled), 10, color, -1)
        cv2.circle(vis_map, (px_scaled, py_scaled), 10, (255, 255, 255), 2)
        
        # 绘制方向箭头
        arrow_len = 18
        end_x = int(px_scaled + arrow_len * np.cos(theta))
        end_y = int(py_scaled - arrow_len * np.sin(theta))  # 注意 y 方向
        cv2.arrowedLine(vis_map, (px_scaled, py_scaled), (end_x, end_y), 
                       (255, 255, 255), 2, tipLength=0.4)
        
        # 添加机器人标签
        label = f"R{i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
        
        # 标签背景
        label_x = px_scaled - text_w // 2
        label_y = py_scaled - 18
        cv2.rectangle(vis_map, (label_x - 2, label_y - text_h - 2), 
                     (label_x + text_w + 2, label_y + 2), color, -1)
        cv2.putText(vis_map, label, (label_x, label_y), 
                   font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 如果有目标点，绘制机器人到目标的连线
        if goal_points and i < len(goal_points):
            gx, gy = goal_points[i][0], goal_points[i][1]
            gx_scaled = int(gx * scale_x)
            gy_scaled = new_h - int(gy * scale_y)
            # 绘制虚线（用点）
            line_color = tuple(max(0, c - 50) for c in color)
            pts_on_line = 15
            for j in range(pts_on_line):
                t = j / pts_on_line
                lx = int(px_scaled + t * (gx_scaled - px_scaled))
                ly = int(py_scaled + t * (gy_scaled - py_scaled))
                if j % 2 == 0:
                    cv2.circle(vis_map, (lx, ly), 2, line_color, -1)
    
    # ===== 3. 添加图例（右上角，超大号版） =====
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1  # 超大字体
    font_thickness = 2
    line_height = 55  # 超大行高
    
    # 计算图例大小
    legend_items = num_agents + 2  # robots + frontier + goal
    legend_w = 260  # 更宽
    legend_h = line_height * (legend_items + 1) + 40  # +1 for title
    
    # 右上角位置
    legend_x = new_w - legend_w - 15
    legend_y = 35
    
    # 背景（半透明效果通过深色实现）
    cv2.rectangle(vis_map, (legend_x - 15, legend_y - 30), 
                 (legend_x + legend_w, legend_y + legend_h), (30, 30, 30), -1)
    cv2.rectangle(vis_map, (legend_x - 15, legend_y - 30), 
                 (legend_x + legend_w, legend_y + legend_h), (150, 150, 150), 2)
    
    # 标题
    cv2.putText(vis_map, f"Step: {step}", (legend_x, legend_y + 10), 
               font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    legend_y += line_height + 10
    
    # 分隔线
    cv2.line(vis_map, (legend_x - 8, legend_y - 15), 
            (legend_x + legend_w - 8, legend_y - 15), (100, 100, 100), 2)
    
    # Robots
    for i in range(num_agents):
        color = robot_colors[i % len(robot_colors)]
        # 超大的圆形标记
        cv2.circle(vis_map, (legend_x + 18, legend_y), 18, color, -1)
        cv2.circle(vis_map, (legend_x + 18, legend_y), 18, (255, 255, 255), 2)
        cv2.putText(vis_map, f"Robot {i}", (legend_x + 48, legend_y + 10), 
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        legend_y += line_height
    
    # Frontiers
    cv2.circle(vis_map, (legend_x + 18, legend_y), 16, (255, 200, 100), -1)
    cv2.circle(vis_map, (legend_x + 18, legend_y), 16, (200, 150, 50), 2)
    cv2.putText(vis_map, "Frontier", (legend_x + 48, legend_y + 10), 
               font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    legend_y += line_height
    
    # Goal（星形）
    star_x, star_y = legend_x + 18, legend_y
    star_size = 18  # 超大星形
    pts = []
    for j in range(5):
        angle_out = -np.pi/2 + j * 2 * np.pi / 5
        pts.append([star_x + int(star_size * np.cos(angle_out)),
                   star_y + int(star_size * np.sin(angle_out))])
        angle_in = angle_out + np.pi / 5
        pts.append([star_x + int(star_size * 0.4 * np.cos(angle_in)),
                   star_y + int(star_size * 0.4 * np.sin(angle_in))])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(vis_map, [pts], (0, 255, 255))
    cv2.polylines(vis_map, [pts], True, (0, 200, 200), 2)
    cv2.putText(vis_map, "Goal", (legend_x + 48, legend_y + 10), 
               font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # ===== 4. 保存图像 =====
    if args.print_images:
        dump_dir = "{}/dump/{}".format(args.dump_location, args.nav_mode)
        ep_dir = '{}/episodes_multi/{}/eps_{}/'.format(dump_dir, rank, episode_n)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + f'TopView_Frontiers-{step}.png'
        cv2.imwrite(fn, vis_map)
    
    if args.visualize:
        cv2.imshow("TopView_Frontiers", vis_map)
        cv2.waitKey(1)
    
    return vis_map


def visualize_agent_rgbd(args, step, observations, episode_n=0, rank=0):
    """
    为每个 agent 保存 RGB + Depth 的可视化图像。
    
    Args:
        args: 命令行参数
        step: 当前步数
        observations: 观测列表，每个元素包含 'rgb' 和 'depth'
        episode_n: episode 编号
        rank: 进程编号
    
    Returns:
        list: 生成的图像数组列表（每个 agent 一张）
    """
    num_agents = len(observations)
    rgbd_images = []
    
    dump_dir = "{}/dump/{}".format(args.dump_location, args.nav_mode)
    ep_dir = '{}/episodes_multi/{}/eps_{}/'.format(dump_dir, rank, episode_n)
    
    if args.print_images:
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
    
    for i, obs in enumerate(observations):
        rgb = obs.get('rgb', None)
        depth = obs.get('depth', None)
        
        if rgb is None:
            continue
        
        # RGB 图像处理
        rgb_vis = rgb.copy()
        if rgb_vis.shape[-1] == 3:
            # 确保是 uint8
            if rgb_vis.dtype != np.uint8:
                rgb_vis = (rgb_vis * 255).astype(np.uint8) if rgb_vis.max() <= 1.0 else rgb_vis.astype(np.uint8)
        
        h, w = rgb_vis.shape[:2]
        
        # Depth 图像处理
        if depth is not None:
            depth_img = depth.copy()
            # 如果 depth 有额外的维度，去掉
            if len(depth_img.shape) == 3:
                depth_img = depth_img.squeeze(-1)
            
            # 归一化 depth 到 0-255
            depth_min = depth_img.min()
            depth_max = depth_img.max()
            if depth_max > depth_min:
                depth_normalized = (depth_img - depth_min) / (depth_max - depth_min) * 255
            else:
                depth_normalized = np.zeros_like(depth_img)
            depth_normalized = depth_normalized.astype(np.uint8)
            
            # 应用 colormap 使深度图更容易可视化
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # 调整 depth 大小以匹配 rgb
            if depth_colored.shape[:2] != (h, w):
                depth_colored = cv2.resize(depth_colored, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 创建 RGB+Depth 并排图像
            # 在两张图之间添加分隔线
            separator_width = 5
            combined_width = w * 2 + separator_width
            combined_image = np.zeros((h, combined_width, 3), dtype=np.uint8)
            
            # RGB 在左边 (转换为 BGR 用于 OpenCV)
            combined_image[:, :w, :] = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
            
            # 分隔线（白色）
            combined_image[:, w:w+separator_width, :] = 255
            
            # Depth 在右边
            combined_image[:, w+separator_width:, :] = depth_colored
            
            # # 添加标签
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 0.6
            # font_thickness = 2
            # text_color = (255, 255, 255)  # 白色
            # bg_color = (0, 0, 0)  # 黑色背景
            
            # # RGB 标签
            # text_rgb = f"Agent {i} - RGB"
            # (text_w, text_h), _ = cv2.getTextSize(text_rgb, font, font_scale, font_thickness)
            # cv2.rectangle(combined_image, (5, 5), (text_w + 10, text_h + 10), bg_color, -1)
            # cv2.putText(combined_image, text_rgb, (7, text_h + 7), font, font_scale, text_color, font_thickness)
            
            # # Depth 标签
            # text_depth = f"Agent {i} - Depth"
            # (text_w, text_h), _ = cv2.getTextSize(text_depth, font, font_scale, font_thickness)
            # cv2.rectangle(combined_image, (w + separator_width + 5, 5), 
            #              (w + separator_width + text_w + 10, text_h + 10), bg_color, -1)
            # cv2.putText(combined_image, text_depth, (w + separator_width + 7, text_h + 7), 
            #            font, font_scale, text_color, font_thickness)
            
            # # 添加步数信息
            # step_text = f"Step: {step}"
            # (text_w, text_h), _ = cv2.getTextSize(step_text, font, font_scale, font_thickness)
            # cv2.rectangle(combined_image, (combined_width - text_w - 15, 5), 
            #              (combined_width - 5, text_h + 10), bg_color, -1)
            # cv2.putText(combined_image, step_text, (combined_width - text_w - 10, text_h + 7), 
            #            font, font_scale, text_color, font_thickness)
        else:
            # 如果没有 depth，只保存 RGB
            combined_image = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
            
            # 添加标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Agent {i} - RGB (No Depth)"
            cv2.putText(combined_image, text, (10, 25), font, 0.6, (255, 255, 255), 2)
        
        rgbd_images.append(combined_image)
        
        # 保存图像
        if args.print_images:
            fn = ep_dir + f'agent-{i}-RGBD-{step}.png'
            cv2.imwrite(fn, combined_image)
        
        # 可视化显示
        if args.visualize:
            cv2.imshow(f"Agent_{i}_RGBD", combined_image)
            cv2.waitKey(1)
    
    return rgbd_images


def Visualize(args, step, pose_pred, map_pred, exp_pred, goal_name, visited_vis, map_edge, goal_map, top_view_map, episode_n=0, rank=0):
    sem_map = np.zeros(map_pred.shape)

    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge >0

    sem_map[exp_mask] = 2
    sem_map[map_mask] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal_map[i], selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 3+i
            
    sem_map[edge_mask] = 3

    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    vis_image = init_multi_vis_image(goal_name, color, 537, 980)

    vis_image[50:530, 15:495] = sem_map_vis
    top_view_map_nor = cv2.resize(top_view_map, (480, 480),
                                interpolation=cv2.INTER_NEAREST)
    vis_image[50:530, 500:980] = np.flipud(top_view_map_nor)

    for i in range(args.num_agents):
        agent_arrow = get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)

    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_{}".format(rank), vis_image)
        cv2.waitKey(1)
    
    if args.print_images:
        dump_dir = "{}/dump/{}".format(args.dump_location, args.nav_mode)
        ep_dir = '{}/episodes_multi/{}/eps_{}/'.format(
            dump_dir, rank, episode_n)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + 'Merged_Vis-{}.png'.format(step)
        cv2.imwrite(fn, vis_image)

    return vis_image