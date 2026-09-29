import bpy
import csv
import math
import os

# 存放 CSV 的根目录（包含多个子文件夹）
csv_root = "F:/图像数据/Area_2/WC_1"

obj_id = 0  # 全局唯一命名计数

# 递归遍历目录及子目录
for root, dirs, files in os.walk(csv_root):
    for file_name in files:
        if file_name.endswith(".csv"):
            csv_path = os.path.join(root, file_name)
            print("正在导入:", csv_path)

            # 创建一个 Collection，以文件名命名
            base_name = os.path.splitext(file_name)[0]
            if base_name not in bpy.data.collections:
                file_collection = bpy.data.collections.new(base_name)
                bpy.context.scene.collection.children.link(file_collection)
            else:
                file_collection = bpy.data.collections[base_name]

            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',')
                for row in reader:
                    cls = row['class']
                    center_x = float(row['center_x'])
                    center_y = float(row['center_y'])
                    center_z = float(row['center_z'])
                    size_x = float(row['size_x'])
                    size_y = float(row['size_y'])
                    size_z = float(row['size_z'])
                    rot_x = math.radians(float(row['angle_x']))
                    rot_y = math.radians(float(row['angle_y']))
                    rot_z = math.radians(float(row['angle_z']))

                    # 添加立方体
                    bpy.ops.mesh.primitive_cube_add(size=1, location=(center_x, center_y, center_z))
                    obj = bpy.context.object

                    # 设置缩放
                    obj.scale = (size_x, size_y, size_z)

                    # 设置旋转
                    obj.rotation_euler = (rot_x, rot_y, rot_z)

                    # 设置名字（带上文件名前缀防止重名）
                    obj.name = f"{base_name}_{cls}_{obj_id}"
                    obj_id += 1

                    # 添加材质
                    mat = bpy.data.materials.get(f"Mat_{cls}")
                    if not mat:
                        mat = bpy.data.materials.new(name=f"Mat_{cls}")
                        mat.diffuse_color = {
                            'beam': (0.5, 0.5, 0.5, 1),
                            'board': (0, 1, 0, 1),
                            'bookcase': (0, 0, 1, 1),
                            'ceiling': (1, 1, 0, 1),
                            'door': (1, 0.5, 0, 1),
                            'floor': (0.5, 0.25, 0, 1),
                            'wall': (0.5, 0.5, 0.5, 1),
                            'column': (0, 1, 1, 1),
                            'window': (1, 0, 1, 1),
                            'chair': (0.7, 0, 0, 1),
                            'table': (0, 0.7, 0, 1),
                            'clutter': (0, 0, 0.7, 1),
                        }.get(cls, (1, 1, 1, 1))  # 默认白色
                        mat.use_nodes = True

                    if len(obj.data.materials) == 0:
                        obj.data.materials.append(mat)

                    # --- 处理子 Collection ---
                    if cls not in file_collection.children:
                        # 创建 class 子 Collection
                        class_collection = bpy.data.collections.new(cls)
                        file_collection.children.link(class_collection)
                    else:
                        class_collection = file_collection.children[cls]

                    # 先移除物体在默认集合中的链接
                    for c in obj.users_collection:
                        c.objects.unlink(obj)

                    # 链接到子 Collection
                    class_collection.objects.link(obj)
