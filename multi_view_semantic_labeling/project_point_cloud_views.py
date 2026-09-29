import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_point_cloud(csv_paths):
    all_xyz = []
    all_rgb = []

    for path in csv_paths:
        data = pd.read_csv(path, skiprows=1, header=None)
        data = data.iloc[:, :6]
        data.columns = ['x', 'y', 'z', 'r', 'g', 'b']
        xyz = data[['x', 'y', 'z']].values.astype(np.float32)
        rgb = data[['r', 'g', 'b']].values.astype(np.uint8)
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    xyz_total = np.concatenate(all_xyz, axis=0)
    rgb_total = np.concatenate(all_rgb, axis=0)
    return xyz_total, rgb_total


def create_camera_extrinsic(eye, target, up=np.array([0, 0, -1])):
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    z_axis = target - eye
    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = -R.T @ eye

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = T

    return extrinsic


def project_points(xyz, rgb, intrinsic, extrinsic, img_size):
    N = xyz.shape[0]
    xyz_h = np.hstack((xyz, np.ones((N, 1))))
    cam_coords = (extrinsic @ xyz_h.T).T[:, :3]
    z = cam_coords[:, 2]
    valid = z > 0
    cam_coords = cam_coords[valid]
    rgb = rgb[valid] / 255.0
    z = z[valid]
    indices = np.arange(N)[valid]

    x, y = cam_coords[:, 0], cam_coords[:, 1]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x_proj = (fx * x / z) + cx
    y_proj = (fy * y / z) + cy

    x_pix = np.round(x_proj).astype(int)
    y_pix = np.round(y_proj).astype(int)

    H, W = img_size
    in_bounds = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)

    return x_pix[in_bounds], y_pix[in_bounds], rgb[in_bounds], z[in_bounds], indices[in_bounds]



def create_image(x_pix, y_pix, colors, depths, indices, img_size):
    H, W = img_size
    image = np.zeros((H, W, 3), dtype=np.float32)
    depth_buffer = np.full((H, W), np.inf)
    index_map = np.zeros((H, W), dtype=np.int32)

    for x, y, color, depth, idx in zip(x_pix, y_pix, colors, depths, indices):
        if depth < depth_buffer[y, x]:
            image[y, x] = color
            depth_buffer[y, x] = depth
            index_map[y, x] = idx + 1  # 背景为0，真实索引从1开始

    return image, index_map



def save_projection(xyz, rgb, intrinsic, img_size, eye, target, save_path):
    extrinsic = create_camera_extrinsic(eye, target)
    x_pix, y_pix, colors, depths, indices = project_points(xyz, rgb, intrinsic, extrinsic, img_size)
    image, index_map = create_image(x_pix, y_pix, colors, depths, indices, img_size)

    # 保存图片
    plt.figure(figsize=(img_size[1] / 100, img_size[0] / 100), dpi=100)
    plt.imshow(image)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 保存 index_map
    #index_map_path = save_path.replace('.png', '_index.npy')
    #np.save(index_map_path, index_map)
    np.savetxt(save_path.replace('.png', '_index.csv'), index_map, fmt='%d', delimiter=',')


def main():
    csv_paths = [
        'C:/Users/Admin/Desktop/point_cloud_part_1.csv',
        'C:/Users/Admin/Desktop/point_cloud_part_2.csv',
        'C:/Users/Admin/Desktop/point_cloud_part_3.csv'
    ]
    img_size = (1080, 1920)

    xyz, rgb = load_point_cloud(csv_paths)
    center = xyz.mean(axis=0)

    fx = 50 * img_size[1] / 36
    fy = 50 * img_size[0] / 24
    cx = img_size[1] / 2
    cy = img_size[0] / 2
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    output_dir = 'C:/Users/Admin/Desktop/projection_circle/'
    os.makedirs(output_dir, exist_ok=True)

    num_views = 36
    radius = 20
    camera_height = center[2] + 20  # 从上往下俯视

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        eye_x = center[0] + radius * np.cos(angle)
        eye_y = center[1] + radius * np.sin(angle)
        eye_z = camera_height
        eye = np.array([eye_x, eye_y, eye_z])
        target = center

        save_path = os.path.join(output_dir, f'view_{i:02d}.png')
        print(f"Rendering view {i + 1}/{num_views}: {save_path}")
        save_projection(xyz, rgb, intrinsic, img_size, eye, target, save_path)

    print("Done. Images saved to:", output_dir)


if __name__ == '__main__':
    main()
