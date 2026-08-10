import math
from typing import Iterable
import dataclasses
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

import numpy as np
from typing import List, Union
import skimage.morphology
from PIL import Image
from constants import color_palette, coco_categories, category_to_id

import supervision as sv
from supervision.draw.color import Color, ColorPalette


def fit_image_to_panel(
    image: np.ndarray,
    panel_width: int,
    panel_height: int,
    *,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize an HWC image to a fixed panel without changing aspect ratio."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("panel image must have shape (height, width, 3)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("panel image dimensions must be positive")
    panel_width = int(panel_width)
    panel_height = int(panel_height)
    if panel_width <= 0 or panel_height <= 0:
        raise ValueError("panel dimensions must be positive")

    scale = min(
        panel_width / float(image.shape[1]),
        panel_height / float(image.shape[0]),
    )
    resized_width = max(1, min(panel_width, int(round(image.shape[1] * scale))))
    resized_height = max(
        1, min(panel_height, int(round(image.shape[0] * scale)))
    )
    interpolation = (
        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    panel = np.full(
        (panel_height, panel_width, 3),
        np.asarray(pad_value, dtype=image.dtype),
        dtype=image.dtype,
    )
    offset_x = (panel_width - resized_width) // 2
    offset_y = (panel_height - resized_height) // 2
    panel[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized
    return panel


# Copied from https://github.com/concept-graphs/concept-graphs/     
def vis_result_fast(
    image: np.ndarray, 
    detections: sv.Detections, 
    classes: List[str], 
    color: Union[Color, ColorPalette] = ColorPalette.default(),
    instance_random_color: bool = False,
    draw_bbox: bool = True,
    mask_exclude_classes: Iterable[str] = ("fire",),
) -> np.ndarray:
    """Annotate detections without hiding rendered fire texture.

    FireWorld supplies a smoke-invariant thermal fire mask for perception.
    Filling that mask with a semantic palette color makes ``--print_images``
    show a flat green blob instead of the underlying volumetric flame. Keep
    its bounding box and label, but reserve mask fills for the other classes.
    """
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
    mask_detections = detections
    excluded_names = {
        str(name).casefold()
        for name in mask_exclude_classes
    }
    excluded_ids = {
        class_id
        for class_id, class_name in enumerate(classes)
        if class_name.casefold() in excluded_names
    }
    if excluded_ids and detections.class_id is not None:
        keep_mask = ~np.isin(
            np.asarray(detections.class_id),
            np.fromiter(excluded_ids, dtype=np.int64),
        )
        mask_detections = detections[keep_mask]
    
    if instance_random_color:
        # Generate random colors for each instance
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
        
    # Apply mask annotations
    annotated_image = image.copy()
    if len(mask_detections) > 0 and mask_detections.mask is not None:
        annotated_image = mask_annotator.annotate(
            scene=annotated_image,
            detections=mask_detections,
        )
    
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


def overlay_hazard_on_obstacle_map(
    map_bgr,
    planning_risk,
    hard_unsafe_mask=None,
    obstacle_mask=None,
    *,
    max_alpha=0.72,
):
    """Blend the planner hazard field onto a rendered obstacle map.

    ``planning_risk`` is the normalized ``[0, 1]`` map consumed by the
    planners. Low values are faint yellow, medium values orange and high
    values red. Hard-unsafe cells receive a magenta fill and outline.
    Physical obstacle pixels are restored after blending so this function is
    strictly a visualization layer and cannot make map semantics ambiguous.
    """

    base = np.asarray(map_bgr)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("map_bgr must have shape (height, width, 3)")
    if base.dtype != np.uint8:
        raise ValueError("map_bgr must use uint8 BGR pixels")

    shape = base.shape[:2]
    risk = np.asarray(planning_risk, dtype=np.float32)
    if risk.shape != shape:
        raise ValueError(
            "planning_risk shape {} does not match obstacle map {}".format(
                risk.shape, shape
            )
        )
    risk = np.clip(
        np.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )

    if hard_unsafe_mask is None:
        hard = np.zeros(shape, dtype=bool)
    else:
        hard = np.asarray(hard_unsafe_mask, dtype=bool)
        if hard.shape != shape:
            raise ValueError(
                "hard_unsafe_mask shape {} does not match obstacle map {}"
                .format(hard.shape, shape)
            )

    if obstacle_mask is None:
        obstacles = np.zeros(shape, dtype=bool)
    else:
        obstacles = np.asarray(obstacle_mask, dtype=bool)
        if obstacles.shape != shape:
            raise ValueError(
                "obstacle_mask shape {} does not match obstacle map {}"
                .format(obstacles.shape, shape)
            )

    result = base.copy()
    heat = np.empty_like(result)
    heat[..., 0] = 0
    heat[..., 1] = np.rint(255.0 * (1.0 - risk)).astype(np.uint8)
    heat[..., 2] = 255
    alpha = (
        np.clip(float(max_alpha), 0.0, 1.0) * risk
    )[..., None]
    blended = (
        result.astype(np.float32) * (1.0 - alpha)
        + heat.astype(np.float32) * alpha
    )
    visible = ~obstacles
    result[visible] = np.rint(blended[visible]).astype(np.uint8)

    hard_visible = hard & visible
    if np.any(hard_visible):
        hard_color = np.asarray([180, 0, 255], dtype=np.float32)
        result[hard_visible] = np.rint(
            0.18 * result[hard_visible].astype(np.float32)
            + 0.82 * hard_color
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            hard_visible.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(result, contours, -1, (80, 0, 160), 1)

    # Contour drawing can touch the neighboring obstacle pixel. Restore every
    # physical obstacle exactly, keeping occupancy and hazard visually distinct.
    result[obstacles] = base[obstacles]
    return result


def draw_hazard_legend(image_bgr, display_label=None):
    """Draw the hazard scale in the navigation panel header."""

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")
    if image.shape[0] < 45 or image.shape[1] < 930:
        return image

    x0, y0, width, height = 600, 8, 150, 10
    values = np.linspace(0.0, 1.0, width, dtype=np.float32)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    gradient[..., 1] = np.rint(
        255.0 * (1.0 - values)
    ).astype(np.uint8)[None, :]
    gradient[..., 2] = 255
    image[y0:y0 + height, x0:x0 + width] = gradient
    cv2.rectangle(
        image, (x0 - 1, y0 - 1), (x0 + width, y0 + height), (40, 40, 40), 1
    )
    cv2.putText(
        image, "Hazard", (535, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (20, 20, 20), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, "low", (600, 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.34, (20, 20, 20), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, "high", (722, 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.34, (20, 20, 20), 1, cv2.LINE_AA,
    )
    cv2.rectangle(image, (805, 8), (821, 20), (180, 0, 255), -1)
    cv2.rectangle(image, (805, 8), (821, 20), (80, 0, 160), 1)
    cv2.putText(
        image, "hard unsafe", (828, 19), cv2.FONT_HERSHEY_SIMPLEX,
        0.34, (20, 20, 20), 1, cv2.LINE_AA,
    )
    if display_label:
        cv2.putText(
            image, str(display_label), (805, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34, (120, 0, 120), 1, cv2.LINE_AA,
        )
    return image


def Visualize(
    args,
    step,
    pose_pred,
    map_pred,
    exp_pred,
    goal_name,
    visited_vis,
    map_edge,
    goal_map,
    top_view_map,
    episode_n=0,
    rank=0,
    planning_risk=None,
    hard_unsafe_mask=None,
    hazard_display_label=None,
):
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
    sem_map_vis = np.asarray(sem_map_vis.convert("RGB"))[:, :, [2, 1, 0]]
    if planning_risk is not None:
        sem_map_vis = overlay_hazard_on_obstacle_map(
            sem_map_vis,
            planning_risk,
            hard_unsafe_mask=hard_unsafe_mask,
            obstacle_mask=map_mask,
        )
    sem_map_vis = np.flipud(sem_map_vis)
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    vis_image = init_multi_vis_image(goal_name, color, 537, 980)
    if planning_risk is not None:
        draw_hazard_legend(vis_image, hazard_display_label)

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
