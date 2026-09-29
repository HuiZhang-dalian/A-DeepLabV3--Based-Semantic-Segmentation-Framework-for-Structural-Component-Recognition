import os
import numpy as np
import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm
import math
import csv
import os
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def visualize_walls(points, results, img_size=(512, 512)):
    """
    可视化点云俯视图 + 检测出的墙体线段
    """
    xy = points[:, [0, 1]]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    scale = max_xy - min_xy

    def world_to_img(x, y):
        x_img = int((x - min_xy[0]) / scale[0] * (img_size[1] - 1))
        y_img = int((y - min_xy[1]) / scale[1] * (img_size[0] - 1))
        return x_img, y_img

    # 背景：点云投影
    plt.figure(figsize=(8, 8))
    plt.scatter(xy[:, 0], xy[:, 1], s=1, c="lightgray", alpha=0.5, label="Points")

    # 墙体：绘制合并后的端点
    for wall in results:
        p1 = wall["p1"]
        p2 = wall["p2"]

        # 画墙体线段
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=2)

        # 标注中心点
        plt.plot(wall["center_x"], wall["center_y"], "bo", markersize=5)
        plt.text(wall["center_x"], wall["center_y"],
                 f"{wall['angle_z']:.1f}°", fontsize=8, color="blue")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Wall Extraction Visualization")
    plt.axis("equal")
    plt.legend()
    plt.show()


def extract_wall_info(points, img_size=(512, 512), voxel_size=0.05,
                      merge_eps_angle=10, merge_eps_dist=0.2,
                      overlap_ratio_thresh=0.1, gap_thresh=0.1):
    xy = points[:, [0, 1]]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    scale = max_xy - min_xy
    norm_x = ((xy[:, 0] - min_xy[0]) / scale[0]) * (img_size[1] - 1)
    norm_y = ((xy[:, 1] - min_xy[1]) / scale[1]) * (img_size[0] - 1)
    norm_xy = np.stack([norm_x, norm_y], axis=1).astype(int)

    canvas = np.zeros(img_size, dtype=np.uint8)
    for x, y in norm_xy:
        if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
            canvas[y, x] = 255

    edges = cv2.Canny(canvas, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=20)
    if lines is None:
        return []

    def get_line_endpoints_from_edges(rho, theta, edges, tolerance=2, gap_thresh=20):
        """
        从边缘图中提取一条直线的端点，带连续性检测，避免误延长到其他墙体。
        """
        a, b = np.cos(theta), np.sin(theta)
        coords = np.column_stack(np.where(edges > 0))  # (y, x)
        selected_points = []
        for y, x in coords:
            rho_point = x * a + y * b
            if abs(rho_point - rho) <= tolerance:
                selected_points.append((x, y))
        if len(selected_points) < 2:
            return None

        selected_points = np.array(selected_points)
        # 投影方向（与直线平行）
        line_dir = np.array([-b, a])
        projections = selected_points @ line_dir
        sort_idx = np.argsort(projections)
        projections_sorted = projections[sort_idx]
        points_sorted = selected_points[sort_idx]

        # === 连续性检测：找到最大簇 ===
        gaps = np.diff(projections_sorted)
        # 找到所有间隙大于阈值的位置
        split_idx = np.where(gaps > gap_thresh)[0] + 1
        clusters = np.split(points_sorted, split_idx)

        # 选择点数最多的簇（假设它对应真正的墙）
        largest_cluster = max(clusters, key=lambda c: len(c))
        if len(largest_cluster) < 2:
            return None

        # 在该簇上找端点
        projections_cluster = largest_cluster @ line_dir
        idx_min, idx_max = projections_cluster.argmin(), projections_cluster.argmax()
        return tuple(largest_cluster[idx_min]), tuple(largest_cluster[idx_max])

    def img_to_world(x, y):
        x_world = x / (img_size[1] - 1) * scale[0] + min_xy[0]
        y_world = y / (img_size[0] - 1) * scale[1] + min_xy[1]
        return x_world, y_world

    # === 1. 收集所有线段信息 ===
    raw_lines = []
    for line in lines:
        rho, theta = line[0]
        endpoints = get_line_endpoints_from_edges(rho, theta, edges)
        if endpoints is None:
            continue
        (x1_img, y1_img), (x2_img, y2_img) = endpoints
        x1_w, y1_w = img_to_world(x1_img, y1_img)
        x2_w, y2_w = img_to_world(x2_img, y2_img)

        cx = (x1_w + x2_w) / 2
        cy = (y1_w + y2_w) / 2
        angle = np.arctan2(y2_w - y1_w, x2_w - x1_w)
        angle_deg = np.degrees(angle) % 180

        raw_lines.append({
            "p1": (x1_w, y1_w),
            "p2": (x2_w, y2_w),
            "center": (cx, cy),
            "angle_deg": angle_deg
        })

    if not raw_lines:
        return []
    #plt.figure(figsize=(6, 6))
    #plt.scatter(points[:, 0], points[:, 1], s=1, c="lightgray")
    #for line in raw_lines:
    #    x1, y1 = line["p1"]
    #    x2, y2 = line["p2"]
    #    plt.plot([x1, x2], [y1, y2], "r-", linewidth=2)
    #plt.title("Raw Hough Lines (before merging)")
    #plt.axis("equal")
    #plt.show()
    # === 🔄 改进 1: 线段–线段最短距离 ===
    def segment_to_segment_distance(p1, p2, q1, q2):
        """计算两条线段之间的最短距离"""
        def point_to_segment_distance(p, a, b):
            ap, ab = p - a, b - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)
            proj = a + t * ab
            return np.linalg.norm(p - proj)

        p1, p2, q1, q2 = map(np.array, [p1, p2, q1, q2])
        return min(
            point_to_segment_distance(p1, q1, q2),
            point_to_segment_distance(p2, q1, q2),
            point_to_segment_distance(q1, p1, p2),
            point_to_segment_distance(q2, p1, p2)
        )

    # === 🔄 改进 2: 投影区间重叠判断 ===
    def overlap_ratio(p1, p2, q1, q2, direction):
        """计算两条线段在指定方向上的投影重叠比例"""
        direction = direction / np.linalg.norm(direction)
        proj_p = sorted([(np.array(p1) @ direction), (np.array(p2) @ direction)])
        proj_q = sorted([(np.array(q1) @ direction), (np.array(q2) @ direction)])
        inter = max(0, min(proj_p[1], proj_q[1]) - max(proj_p[0], proj_q[0]))
        len_p = proj_p[1] - proj_p[0]
        len_q = proj_q[1] - proj_q[0]
        if len_p < 1e-6 or len_q < 1e-6:
            return 0
        return inter / min(len_p, len_q)  # 用较短的线段作为归一化基准

    # === 🔄 改进 3: 沿方向平移距离 ===
    def parallel_gap(p1, p2, q1, q2, direction):
        """计算两条线段在方向上的平移间隔（投影区间之间的间距）"""
        direction = direction / np.linalg.norm(direction)
        proj_p = sorted([(np.array(p1) @ direction), (np.array(p2) @ direction)])
        proj_q = sorted([(np.array(q1) @ direction), (np.array(q2) @ direction)])
        if proj_p[1] < proj_q[0]:
            return proj_q[0] - proj_p[1]
        elif proj_q[1] < proj_p[0]:
            return proj_p[0] - proj_q[1]
        else:
            return 0.0

    # === 2. 构建图，判断哪些线段可合并 ===
    G = nx.Graph()
    for i, line_i in enumerate(raw_lines):
        G.add_node(i)
        for j in range(i + 1, len(raw_lines)):
            line_j = raw_lines[j]
            angle_diff = abs(line_i['angle_deg'] - line_j['angle_deg'])
            angle_diff = min(angle_diff, 180 - angle_diff)
            if angle_diff > merge_eps_angle:
                continue

            # 线段最短距离
            dist = segment_to_segment_distance(line_i['p1'], line_i['p2'],
                                               line_j['p1'], line_j['p2'])
            if dist > merge_eps_dist:
                continue

            # 投影重叠比例
            main_dir = np.array([np.cos(np.radians(line_i['angle_deg'])),
                                 np.sin(np.radians(line_i['angle_deg']))])
            overlap = overlap_ratio(line_i['p1'], line_i['p2'],
                                    line_j['p1'], line_j['p2'], main_dir)
            if overlap < overlap_ratio_thresh:
                continue

            # 沿方向的平移距离
            gap = parallel_gap(line_i['p1'], line_i['p2'],
                               line_j['p1'], line_j['p2'], main_dir)
            if gap > gap_thresh:
                continue

            # 若满足所有条件，则认为两线可合并
            G.add_edge(i, j)

    # === 3. 每个连通子图合并为一堵墙 ===
    results = []
    for component in nx.connected_components(G):
        group = [raw_lines[i] for i in component]
        all_points = [line["p1"] for line in group] + [line["p2"] for line in group]
        all_points = np.array(all_points)

        pca = PCA(n_components=2)
        pca.fit(all_points)
        direction = pca.components_[0]
        projections = all_points @ direction
        p1 = all_points[projections.argmin()]
        p2 = all_points[projections.argmax()]

        cx = (p1[0] + p2[0]) / 2
        cy = (p1[1] + p2[1]) / 2
        cz = (points[:, 2].max() + points[:, 2].min()) / 2
        length = np.linalg.norm(p2 - p1)
        thickness = voxel_size
        height = points[:, 2].max() - points[:, 2].min()
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        angle_deg = np.degrees(angle) % 180

        results.append({
            "class": "wall",
            "center_x": cx,
            "center_y": cy,
            "center_z": cz,
            "size_x": length,
            "size_y": thickness,
            "size_z": height,
            "angle_x": 0,
            "angle_y": 0,
            "angle_z": angle_deg,
            #"p1": p1.tolist(),
            #"p2": p2.tolist()
        })

    return results


def visualize_components(original_points, components_info):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色点云

    geometries = [pcd]

    for comp in components_info:
        center = np.array([comp['center_x'], comp['center_y'], comp['center_z']])
        size = np.array([comp['size_x'], comp['size_y'], comp['size_z']])
        angles = np.radians([comp['angle_x'], comp['angle_y'], comp['angle_z']])
        R = euler_to_rotation_matrix(*angles)
        obb = o3d.geometry.OrientedBoundingBox(center, R, size)

        color = {
            "wall": [1, 0, 0],
            "door": [0, 1, 0],
            "window": [0, 0, 1],
            "beam": [1, 1, 0],
            "floor": [0.5, 0.5, 1],
            "ceiling": [1, 0.5, 1],
            "column": [0.2, 0.8, 0.8]
        }.get(comp['class'], [0.8, 0.8, 0.8])
        obb.color = color
        geometries.append(obb)

    o3d.visualization.draw_geometries(geometries)


def euler_to_rotation_matrix(rx, ry, rz):
    # 欧拉角转旋转矩阵（XYZ顺序）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


def load_point_cloud1(file_path):
    data = np.loadtxt(file_path)
    xyz = data[:, :3]
    rgb = data[:, 3:6]
    labels = data[:, 6].astype(int)
    return xyz, rgb, labels




def load_point_cloud(file_path):
    # 1. 忽略列数不对的行，只拿前 6 列
    df = pd.read_csv(file_path, sep=r'\s+', header=None,
                     on_bad_lines='skip', usecols=range(6))
    # 2. 拆成 xyz, rgb, labels
    xyz   = df.iloc[:, :3].values
    rgb   = df.iloc[:, 3:6].values
    # 如果你原来第 7 列是 label，而这里只有 6 列，
    # 说明 label 在另一列，请把 usecols=range(7) 并对应调整
    # 下面先用 0 占位，你按实际改
    labels = df.iloc[:, -1].astype(int).values if df.shape[1] > 3 else np.zeros(len(df), dtype=int)
    return xyz, rgb, labels


def rotation_matrix_to_euler_xyz(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])


def extract_obb_components(points, label_name, eps=0.3, min_samples=30):
    # 先随机下采样到 20 万点
    if points.shape[0] > 200_000:
        idx = np.random.choice(points.shape[0], 200_000, replace=False)
        points = points[idx]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    results = []

    for i in np.unique(labels):
        if i == -1:
            continue
        part = points[labels == i]
        if len(part) < 10:
            continue

        # 在 XYZ 方向直接计算 AABB
        min_vals = np.min(part, axis=0)  # [min_x, min_y, min_z]
        max_vals = np.max(part, axis=0)  # [max_x, max_y, max_z]
        center = (min_vals + max_vals) / 2
        size = max_vals - min_vals

        results.append({
            "class": label_name,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
            "angle_x": 0,
            "angle_y": 0,
            "angle_z": 0,  # 无需旋转
        })

    return results



def compute_z_rotation_angle(R):
    # R 是 PCA.components_，第一行表示最大主方向
    dir_x, dir_y = R[0, 0], R[0, 1]
    angle_rad = math.atan2(dir_y, dir_x)
    angle_deg = math.degrees(angle_rad)
    return ((angle_deg + 180) % 360) - 180  # 映射到 [-180, 180]


def extract_all_components(txt_path, output_csv):
    xyz, _, labels = load_point_cloud(txt_path)
    all_results = []

    for label_id, name in LABELS.items():
        mask = labels == label_id
        points = xyz[mask]
        if len(points) < 100:
            continue
        if label_id == WALL_LABEL:
            results = extract_wall_info(points)
            #visualize_walls(points, results)
        else:
            results = extract_obb_components(points, name)
        all_results.extend(results)
    if not all_results:
        print(f"[!] 未发现任何有效组件，跳过写入：{output_csv}")
        return
    with open(output_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"[✓] 提取完成，输出保存到 {output_csv}")
    # 可视化
    visualize_components(xyz, all_results)


def merge_annotation_files(annotation_dir, save_path):
    """
    合并Annotations目录下的所有点云文件，并根据预设类别名称分配固定整数标签。

    Args:
        annotation_dir (str): Annotations目录路径。
        save_path (str): 输出合并后txt的路径。
    """
    # 固定类别 -> 标签索引
    predefined_labels = {
        "beam": 0,
        "board": 1,
        "bookcase": 2,
        "ceiling": 3,
        "chair": 4,
        "clutter": 5,
        "door": 6,
        "floor": 7,
        "table": 8,
        "wall": 9,
        "window": 10,
        "column": 11,
    }

    data_list = []
    unknown_labels = set()

    for file in sorted(os.listdir(annotation_dir)):
        if not file.endswith('.txt'):
            continue

        filepath = os.path.join(annotation_dir, file)
        class_name = '_'.join(file.split('_')[:-1])  # 例如 beam_1.txt -> beam

        if class_name not in predefined_labels:
            unknown_labels.add(class_name)
            continue  # 跳过未定义的类

        label = predefined_labels[class_name]
        points = pd.read_csv(filepath, sep=r'\s+', header=None,
                             on_bad_lines='skip', usecols=range(6)).values
        labels = np.full((points.shape[0], 1), label)
        points_with_label = np.hstack((points, labels))  # (N, 7)

        data_list.append(points_with_label)

    if data_list:
        merged_data = np.vstack(data_list)
        # 1. 去掉含 NaN/inf 的行
        merged_data = merged_data[np.isfinite(merged_data).all(axis=1)]

        # 2. 再把剩下的 NaN 兜底填 0（防止万一）
        merged_data = np.nan_to_num(merged_data, nan=0.0, posinf=0, neginf=0)
        np.savetxt(save_path, merged_data, fmt='%.6f %.6f %.6f %d %d %d %d')
        print(f"保存成功: {save_path}")
        print(f"标签映射: {predefined_labels}")
        if unknown_labels:
            print(f"⚠️ 未知类别已被跳过: {sorted(unknown_labels)}")
    else:
        print("❌ 目录中未找到有效的 .txt 文件或没有匹配的类别")

if __name__ == '__main__':
    root = r'F:/图像数据/Area_5/lobby_1'
    folder_names = [name for name in os.listdir(root)
                    if os.path.isdir(os.path.join(root, name))]

    # 常量放外面
    LABELS = {
        0: "beam", 3: "ceiling", 6: "door", 7: "floor",
        9: "wall", 10: "window", 11: "column"
    }
    WALL_LABEL = 9

    for nam in folder_names:
        annotation_dir = os.path.join(root, nam, 'Annotations')
        save_path = os.path.join(root, nam, 'merged_points.txt')
        input_txt = save_path
        output_csv = os.path.join(root, nam, 'components_info.csv')

        # 如果子文件夹里不一定有 Annotations，可加判断
        if not os.path.exists(annotation_dir):
            continue

        merge_annotation_files(annotation_dir, save_path)
        extract_all_components(input_txt, output_csv)
