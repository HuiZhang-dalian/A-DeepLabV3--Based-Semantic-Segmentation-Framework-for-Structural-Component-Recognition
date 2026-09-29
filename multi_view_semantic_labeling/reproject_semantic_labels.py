import pandas as pd
import numpy as np
import os
import re
from collections import Counter


def map_point_to_label(point_filename):
    """
    将点序号文件名映射到对应的标签文件名
    例如: "view_00_index.csv" -> "segmentation_view_00_pixel_labels.csv"
    """
    # 使用正则表达式提取视图编号
    match = re.search(r'view_(\d+)_index\.csv', point_filename)
    if match:
        view_num = match.group(1)
        return f"segmentation_view_{view_num}_pixel_labels.csv"
    else:
        # 如果文件名格式不匹配，返回None
        return None


def map_label_to_point(label_filename):
    """
    将标签文件名映射到对应的点序号文件名
    例如: "segmentation_view_00_pixel_labels.csv" -> "view_00_index.csv"
    """
    # 使用正则表达式提取视图编号
    match = re.search(r'segmentation_view_(\d+)_pixel_labels\.csv', label_filename)
    if match:
        view_num = match.group(1)
        return f"view_{view_num}_index.csv"
    else:
        # 如果文件名格式不匹配，返回None
        return None


def process_single_pair(label_csv_path, point_csv_path, view_id=None):
    """
    处理单个CSV文件对，返回处理后的DataFrame
    """

    # 读取CSV文件
    label_df = pd.read_csv(label_csv_path)
    point_df = pd.read_csv(point_csv_path)

    # 检查两个表格的形状是否相同
    if label_df.shape != point_df.shape:
        print(f"  警告: 两个表格形状不同 - 标签表: {label_df.shape}, 点序号表: {point_df.shape}")
        # 取较小的形状以确保对应关系
        min_rows = min(label_df.shape[0], point_df.shape[0])
        min_cols = min(label_df.shape[1], point_df.shape[1])
        label_df = label_df.iloc[:min_rows, :min_cols]
        point_df = point_df.iloc[:min_rows, :min_cols]
        print(f"  调整后形状: {label_df.shape}")

    # 将数据展平为一维数组
    labels_flat = label_df.values.flatten()
    points_flat = point_df.values.flatten()

    # 创建临时DataFrame
    temp_df = pd.DataFrame({
        'point_id': points_flat,
        'label': labels_flat
    })

    print(f"  原始数据点数: {len(temp_df)}")

    # 使用agg函数处理重复点，对每个点的标签取众数
    def get_mode(series):
        mode_values = series.mode()
        if not mode_values.empty:
            return mode_values.iloc[0]  # 如果有多个众数，取第一个
        else:
            return series.iloc[0] if not series.empty else -1

    # 分组聚合
    result_df = temp_df.groupby('point_id')['label'].agg(get_mode).reset_index()

    # 检查重复处理情况
    original_count = len(temp_df)
    final_count = len(result_df)
    if original_count > final_count:
        print(f"  处理了 {original_count - final_count} 个重复点序号")

    # 添加视图ID
    if view_id is not None:
        result_df['view_id'] = view_id

    # 排序
    result_df = result_df.sort_values('point_id').reset_index(drop=True)

    # 显示统计信息
    print(f"  最终数据点数: {len(result_df)}")
    print(f"  点序号范围: {result_df['point_id'].min()} - {result_df['point_id'].max()}")
    print(f"  标签种类: {result_df['label'].nunique()}")

    return result_df


def get_mode_with_tiebreak(series):
    """
    获取众数，处理并列情况

    参数:
        series: pandas Series，包含一个点在多个视图中的标签

    返回:
        最终标签值
    """
    if len(series) == 0:
        return -1  # 空序列返回-1

    # 使用mode()方法获取众数
    mode_values = series.mode()

    if len(mode_values) == 0:
        # 理论上不会发生，但为了安全
        return series.iloc[0]
    elif len(mode_values) == 1:
        # 只有一个众数，直接返回
        return mode_values.iloc[0]
    else:
        # 多个众数（并列），需要解决冲突
        # 策略1: 取最小的标签值（通常更保守）
        # 策略2: 取最近视图的标签（需要额外信息）
        # 策略3: 随机选择一个
        # 这里采用策略1：取最小的标签值
        print(f"  警告: 点ID {series.name} 有 {len(mode_values)} 个并列众数: {mode_values.tolist()}")
        return mode_values.min()


def merge_all_csv_pairs_with_voting(label_folder, point_folder, output_file):
    """
    将所有CSV文件对合并成一个大的CSV文件，使用投票机制解决多视图间的标签冲突
    """

    # 创建输出文件夹（如果路径包含文件夹）
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 获取两个文件夹中的CSV文件列表
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.csv')])
    point_files = sorted([f for f in os.listdir(point_folder) if f.endswith('.csv')])

    print(f"标签文件夹: {len(label_files)} 个文件")
    print(f"点序号文件夹: {len(point_files)} 个文件")

    # 存储所有处理结果的列表
    all_results = []
    processed_pairs = 0
    missing_points = []

    # 处理每个标签文件对
    print("\n开始处理文件对:")
    print("-" * 50)

    for label_file in label_files:
        point_file = map_label_to_point(label_file)

        if point_file is None:
            print(f"跳过无法匹配的标签文件: {label_file}")
            continue

        label_path = os.path.join(label_folder, label_file)
        point_path = os.path.join(point_folder, point_file)

        if not os.path.exists(point_path):
            missing_points.append((label_file, point_file))
            print(f"✗ 找不到对应的点序号文件: {label_file} -> {point_file}")
            continue

        # 提取视图编号
        match = re.search(r'view_(\d+)', label_file)
        view_id = match.group(1) if match else "unknown"

        print(f"\n处理: {label_file} + {point_file} (视图 {view_id})")

        try:
            # 处理单个文件对
            result_df = process_single_pair(label_path, point_path, view_id)
            all_results.append(result_df)
            processed_pairs += 1
            print(f"✓ 成功处理")

        except Exception as e:
            print(f"✗ 处理失败: {label_file} + {point_file}, 错误: {e}")

    # 检查是否成功处理了任何文件对
    if not all_results:
        print("没有成功处理任何文件对！")
        return None

    print(f"\n合并 {len(all_results)} 个文件的结果...")

    # 合并所有结果
    merged_df = pd.concat(all_results, ignore_index=True)
    print(f"合并后的总记录数: {len(merged_df)}")

    # 统计每个点出现的视图数量
    view_counts = merged_df.groupby('point_id')['view_id'].nunique()
    print(f"\n视图覆盖统计:")
    print(f"  平均每个点出现在 {view_counts.mean():.2f} 个视图中")
    print(f"  最多出现在 {view_counts.max()} 个视图中")
    print(f"  最少出现在 {view_counts.min()} 个视图中")

    # 统计有多少点出现在多个视图中
    multi_view_points = (view_counts > 1).sum()
    print(f"  有 {multi_view_points} 个点出现在多个视图中")

    # 使用投票机制解决多视图间的标签冲突
    print("\n进行多视图标签投票...")

    # 按point_id分组，对标签进行投票
    final_df = merged_df.groupby('point_id').agg({
        'label': lambda x: get_mode_with_tiebreak(x),
        'view_id': lambda x: ','.join(sorted(set(x.astype(str))))
    }).reset_index()

    # 重命名列
    final_df.columns = ['point_id', 'label', 'source_views']

    # 按点序号排序
    final_df = final_df.sort_values('point_id').reset_index(drop=True)

    # 保存合并结果
    final_df.to_csv(output_file, index=False)

    # 输出最终统计信息
    print(f"\n{'=' * 60}")
    print("合并完成!")
    print(f"成功处理: {processed_pairs} 对文件")
    print(f"总数据点数: {len(final_df)}")
    print(f"点序号范围: {final_df['point_id'].min()} - {final_df['point_id'].max()}")
    print(f"标签种类: {final_df['label'].nunique()}")
    print(f"输出文件: {output_file}")

    # 显示标签分布
    print(f"\n标签分布:")
    label_counts = final_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"  标签 {label}: {count} 个点 ({percentage:.2f}%)")

    # 显示视图覆盖情况
    print(f"\n视图覆盖详情:")
    source_counts = final_df['source_views'].apply(lambda x: len(x.split(',')))
    source_stats = source_counts.value_counts().sort_index()
    for count, num_points in source_stats.items():
        percentage = (num_points / len(final_df)) * 100
        print(f"  出现在 {count} 个视图中的点: {num_points} 个 ({percentage:.2f}%)")

    if missing_points:
        print(f"\n缺少点序号文件的标签文件 ({len(missing_points)} 个):")
        for label_file, point_file in missing_points:
            print(f"  {label_file} -> 需要 {point_file}")

    return final_df


def analyze_label_conflicts(merged_df_before_voting, final_df_after_voting):
    """
    分析标签冲突的解决情况
    """
    print("\n标签冲突分析:")
    print("-" * 40)

    # 找出在多个视图中出现的点
    multi_view_points = merged_df_before_voting.groupby('point_id').filter(lambda x: len(x) > 1)

    if len(multi_view_points) == 0:
        print("没有发现多视图冲突的点")
        return

    conflict_points = []

    for point_id, group in multi_view_points.groupby('point_id'):
        unique_labels = group['label'].unique()
        if len(unique_labels) > 1:
            # 这个点在多个视图中被赋予了不同的标签
            final_label = final_df_after_voting[final_df_after_voting['point_id'] == point_id]['label'].iloc[0]
            conflict_points.append({
                'point_id': point_id,
                'view_labels': dict(zip(group['view_id'], group['label'])),
                'final_label': final_label,
                'num_views': len(group),
                'num_labels': len(unique_labels)
            })

    if conflict_points:
        print(f"发现 {len(conflict_points)} 个有标签冲突的点")
        print("\n前10个冲突点的解决情况:")
        for i, point in enumerate(conflict_points[:10]):
            print(f"  点ID {point['point_id']}:")
            print(f"    出现在 {point['num_views']} 个视图中，有 {point['num_labels']} 种不同标签")
            print(f"    各视图标签: {point['view_labels']}")
            print(f"    投票后标签: {point['final_label']}")
            print()

        # 统计冲突解决情况
        resolved_stats = {}
        for point in conflict_points:
            # 检查最终标签是否是最常见的
            labels = list(point['view_labels'].values())
            label_counts = Counter(labels)
            most_common = label_counts.most_common(1)[0][0]

            if point['final_label'] == most_common:
                resolved_stats['correct'] = resolved_stats.get('correct', 0) + 1
            else:
                resolved_stats['incorrect'] = resolved_stats.get('incorrect', 0) + 1

        print(f"冲突解决统计:")
        print(f"  符合多数意见: {resolved_stats.get('correct', 0)} 个点")
        print(f"  不符合多数意见: {resolved_stats.get('incorrect', 0)} 个点")
    else:
        print("没有发现标签冲突的点")


def list_available_pairs(label_folder, point_folder):
    """
    列出所有可匹配的文件对
    """
    print("可匹配的文件对:")
    print("-" * 50)

    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.csv')])
    available_pairs = 0

    for label_file in label_files:
        point_file = map_label_to_point(label_file)
        if point_file:
            point_path = os.path.join(point_folder, point_file)
            if os.path.exists(point_path):
                status = "✓ 存在"
                available_pairs += 1
            else:
                status = "✗ 缺失"
            print(f"{label_file}")
            print(f"  -> {point_file} {status}")
            print()

    print(f"总计: {available_pairs} 对可处理的文件")
    return available_pairs


# 使用示例
if __name__ == "__main__":
    # 请替换为您的实际文件夹路径
    label_folder = r"C:\Users\Admin\Desktop\room\output"  # 标签CSV文件夹
    point_folder = r"C:\Users\Admin\Desktop\room"  # 点序号CSV文件夹
    output_file = r"C:\Users\Admin\Desktop\room\merged_all_data_voting.csv"  # 输出文件

    # 首先列出可匹配的文件对
    print("扫描文件匹配情况...")
    available_pairs = list_available_pairs(label_folder, point_folder)

    if available_pairs == 0:
        print("没有找到可匹配的文件对，请检查文件夹路径和文件名！")
    else:
        print("\n" + "=" * 80 + "\n")

        # 开始处理并合并所有文件对
        print("开始处理数据...")
        merged_data = merge_all_csv_pairs_with_voting(label_folder, point_folder, output_file)

        if merged_data is not None:
            # 显示合并后的前几行数据
            print(f"\n合并文件的前10行数据:")
            print("-" * 40)
            print(merged_data.head(10))

            # 显示合并后的后几行数据
            print(f"\n合并文件的后10行数据:")
            print("-" * 40)
            print(merged_data.tail(10))

            # 显示文件大小
            file_size = os.path.getsize(output_file)
            print(f"\n输出文件大小: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")

            # 显示数据基本信息
            print(f"\n数据基本信息:")
            print(f"  总点数: {len(merged_data)}")
            print(f"  点ID范围: {merged_data['point_id'].min()} - {merged_data['point_id'].max()}")
            print(f"  标签数量: {merged_data['label'].nunique()}")

            # 检查是否有无效标签
            invalid_labels = merged_data[merged_data['label'] == -1]
            if len(invalid_labels) > 0:
                print(f"  警告: 发现 {len(invalid_labels)} 个无效标签(-1)")

            # 保存统计信息到文件
            stats_file = output_file.replace('.csv', '_stats.txt')
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("数据合并统计报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"处理时间: {pd.Timestamp.now()}\n")
                f.write(f"标签文件夹: {label_folder}\n")
                f.write(f"点序号文件夹: {point_folder}\n")
                f.write(f"输出文件: {output_file}\n\n")
                f.write(f"总点数: {len(merged_data)}\n")
                f.write(f"点ID范围: {merged_data['point_id'].min()} - {merged_data['point_id'].max()}\n")
                f.write(f"标签种类: {merged_data['label'].nunique()}\n\n")

                f.write("标签分布:\n")
                label_counts = merged_data['label'].value_counts().sort_index()
                for label, count in label_counts.items():
                    percentage = (count / len(merged_data)) * 100
                    f.write(f"  标签 {label}: {count} 个点 ({percentage:.2f}%)\n")

                f.write("\n视图覆盖情况:\n")
                source_counts = merged_data['source_views'].apply(lambda x: len(x.split(',')))
                source_stats = source_counts.value_counts().sort_index()
                for count, num_points in source_stats.items():
                    percentage = (num_points / len(merged_data)) * 100
                    f.write(f"  出现在 {count} 个视图中的点: {num_points} 个 ({percentage:.2f}%)\n")

            print(f"统计报告已保存到: {stats_file}")