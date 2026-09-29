import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import datetime
import torch.utils.model_zoo as model_zoo
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.transforms import functional as TF
import random
import math
# bugai resnet(jiyu4,gpu:2) 数据增强    基础模型


# -----------------------------------------------
# 定义 ASPP 模块
# -----------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        """
        参数:
        - in_channels: 输入特征通道数
        - out_channels: 每个分支卷积输出通道数
        - atrous_rates: 空洞卷积率列表 (例如: [6, 12, 18])
        """
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积分支
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ))
        # 采用不同空洞率的 3x3 卷积分支
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ))
        # 图像级全局平均池化分支
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ))
        self.convs = nn.ModuleList(modules)

        # 融合各分支
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Dropout2d(0.3)
        )

    def forward(self, x):
        _res = []
        for conv in self.convs[:-1]:
            _res.append(conv(x))
        # 对全局池化分支，先计算再上采样到原尺寸
        pool = self.convs[-1](x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        _res.append(pool)
        x = torch.cat(_res, dim=1)
        x = self.project(x)
        return x


# -----------------------------------------------
# 定义 DeepLabV3+ 解码器模块
# -----------------------------------------------
class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels, num_classes, decoder_channels=256):
        """
        参数:
        - low_level_channels: 骨干网络低层特征通道数（用于细节补充）
        - num_classes: 分割的类别数
        - decoder_channels: 解码器中间通道数
        """
        super(DeepLabV3PlusDecoder, self).__init__()
        # 对低层特征进行 1x1 投影降维
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        # 解码器融合模块
        self.decoder = nn.Sequential(
            nn.Conv2d(48 + decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        # 最后预测分割结果
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, x, low_level_feat):
        # 提取低层特征
        low_level_feat = self.conv_low(low_level_feat)
        # 将 ASPP 模块得到的高层特征上采样至低层特征尺寸
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        # 拼接融合
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        return x


# -----------------------------------------------
# 定义 DeepLabV3+ 模型整体
# -----------------------------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained_backbone=True):
        """
        参数:
        - num_classes: 分割类别数（ADE20K 中有150类）
        - backbone: 选择的主干网络
        - pretrained_backbone: 是否加载预训练的骨干网络
        """
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'resnet50':
            resnet = models.resnet101(pretrained=pretrained_backbone)
            # 提取第一个模块（不需要全连接层）
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,  # 低层特征 提供给解码器
                resnet.layer2,
                resnet.layer3,
                resnet.layer4  # 高层特征
            )
            # 低层特征：resnet.layer1 的输出通道数（一般为256）
            low_level_channels = 256
            # 高层特征输出通道数，resnet.layer4 的输出一般为2048
            high_level_channels = 2048
        else:
            raise NotImplementedError('目前仅支持 resnet50 作为 backbone')

        # 使用 ASPP 模块对高层特征处理
        self.aspp = ASPP(in_channels=high_level_channels, out_channels=256, atrous_rates=[6, 12, 18])
        # 解码器模块
        self.decoder = DeepLabV3PlusDecoder(low_level_channels=low_level_channels, num_classes=num_classes,
                                            decoder_channels=256)

    def forward(self, x):
        # 提取各层特征
        # 分为低层和高层特征：这里以 layer1 为低层特征, 后续层作为高层特征
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool
        low_level_feat = self.backbone[4](x)  # layer1 作为低层特征

        x = self.backbone[5](low_level_feat)  # layer2
        x = self.backbone[6](x)  # layer3
        high_level_feat = self.backbone[7](x)  # layer4

        # ASPP 处理
        x = self.aspp(high_level_feat)
        # 解码器融合
        x = self.decoder(x, low_level_feat)
        # 最后上采样到与输入图像相同的尺寸
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

def is_image_file(fname):
    return any(fname.lower().endswith(ext) for ext in IMG_EXT)

class ADE20KDataset(Dataset):
    def __init__(self, root, image_folder='images', mask_folder='masks',
                 base_size=512, crop_size=(512, 512), scale_range=(0.5, 2.0)):
        self.image_dir = os.path.join(root, image_folder)
        self.mask_dir = os.path.join(root, mask_folder)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if is_image_file(f)])
        self.mask_files  = sorted([f for f in os.listdir(self.mask_dir)  if is_image_file(f)])

        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_range = scale_range

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_files)

    def random_resize(self, image, mask):
        """随机缩放，长边 ∈ [base_size*min, base_size*max]"""
        long_size = random.randint(
            int(self.base_size * self.scale_range[0]),
            int(self.base_size * self.scale_range[1])
        )
        w, h = image.size
        if h > w:
            new_h, new_w = long_size, int(w * long_size / h)
        else:
            new_h, new_w = int(h * long_size / w), long_size

        image = TF.resize(image, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask

    def pad_if_needed(self, img, target_size, fill=0):
        w, h = img.size
        pad_h = max(target_size[1] - h, 0)  # target_size 是 (H, W)
        pad_w = max(target_size[0] - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=fill)
        return img

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # 1. 随机缩放
        image, mask = self.random_resize(image, mask)

        # 2. 如果比 crop 小，先 pad
        image = self.pad_if_needed(image, self.crop_size, fill=0)
        mask = self.pad_if_needed(mask, self.crop_size, fill=0)  # 255 作为 ignore_index

        # 3. 随机裁剪固定大小
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # 4. 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 5. 转 tensor + normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = TF.pil_to_tensor(mask).squeeze(0).long()

        return image, mask


class valDataset(Dataset):
    def __init__(self, root, image_folder='images', mask_folder='masks', crop_size=(512, 512)):
        self.image_dir = os.path.join(root, image_folder)
        self.mask_dir = os.path.join(root, mask_folder)
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        self.crop_size = crop_size

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = TF.resize(image, self.crop_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  self.crop_size, interpolation=TF.InterpolationMode.NEAREST)

        # 转 tensor
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = TF.pil_to_tensor(mask).squeeze(0).long()

        return image, mask


def get_data_loaders(root_dir, batch_size=4, crop_size=(512, 512), image_folder='images', mask_folder='masks'):
    dataset = ADE20KDataset(root=root_dir,
                            image_folder=image_folder,
                            mask_folder=mask_folder,
                            crop_size=crop_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return data_loader


def get_valdata_loaders(root_dir, batch_size=4, crop_size=(256, 256), image_folder='images', mask_folder='masks'):
    dataset = valDataset(root=root_dir,
                            image_folder=image_folder,
                            mask_folder=mask_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return data_loader


def calculate_accuracy(output, target):
    """
    计算准确率，忽略无效标签 (0)
    参数:
    - output: 模型输出，大小为 [batch_size, num_classes, height, width]
    - target: 真实标签，大小为 [batch_size, height, width]
    """
    # 获取预测结果的类别索引
    _, predicted = torch.max(output, 1)
    # 忽略值为 0 的无效标签
    mask = target != 0
    correct = (predicted == target) & mask
    accuracy = correct.sum().float() / mask.sum().float()  # 计算准确率
    return accuracy


def calculate_recall(output, target, device, num_classes=150):
    """
    计算每个类别的召回率 (Recall = TP / (TP + FN))，忽略无效标签 0

    参数:
    - output: 模型输出，[B, C, H, W]
    - target: 真实标签，[B, H, W]
    - num_classes: 类别数（含无效标签 0）

    返回:
    - recall_per_class: 每个类别的召回率，[num_classes]
    - tp_per_class: TP 数量，[num_classes]
    - fn_per_class: FN 数量，[num_classes]
    """
    _, pred = torch.max(output, 1)          # [B, H, W]
    pred = pred.to(device)
    target = target.to(device)

    tp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    # 有效掩码：跳过 ignore_index = 0
    valid_mask = (target != 0)
    for c in range(num_classes):
        # 真实类别 c 的像素（跳过 ignore）
        gt_c = (target == c + 1) & valid_mask
        # 预测类别 c 的像素
        pred_c = (pred == c + 1) & valid_mask

        tp[c] = (gt_c & pred_c).sum().float()
        fn[c] = (gt_c & (~pred_c)).sum().float()

    recall_per_class = tp / (tp + fn + 1e-6)
    return recall_per_class


def calculate_iou(output, target, device, num_classes=150):
    """
    计算每个类别的交并比 (IoU)，忽略无效标签 (0)

    参数:
    - output: 模型输出，大小为 [batch_size, num_classes, height, width]
    - target: 真实标签，大小为 [batch_size, height, width]
    - num_classes: 类别数（包括无效标签0）

    返回:
    - iou_per_class: 每个类别的IoU，大小为 [num_classes]
    """
    # 获取预测结果的类别索引
    _, predicted = torch.max(output, 1)
    predicted = predicted.to(device)
    # 初始化每个类别的交集和并集
    target = target.to(device)
    intersection = torch.zeros(num_classes).float().to(device)
    union = torch.zeros(num_classes).float().to(device)

    # 遍历所有类别进行交并比计算
    for c in range(num_classes):
        # 创建二值掩码，只有在类别 c 的位置为1，其余为0
        pred_c = (predicted == c+1)
        target_c = (target == c+1)

        # 计算交集和并集
        intersection[c] = (pred_c & target_c).sum().float()
        union[c] = (pred_c | target_c).sum().float()

    # 计算每个类别的IoU
    iou_per_class = intersection / (union + 1e-6)  # 避免除以0的情况
    return iou_per_class, intersection, union


def mean_iou(preds, labels, device, num_classes, ignore_index=None):
    """
    计算 Mean IoU

    参数:
    - preds: 预测结果 [batch_size, H, W] 或 [batch_size, num_classes, H, W]（若是 logits，会自动转 argmax）
    - labels: 实际标签 [batch_size, H, W]
    - num_classes: 类别数
    - ignore_index: 忽略的标签（如背景类别 0），默认不忽略
    - device: 设备 ('cuda' or 'cpu')

    返回:
    - mIoU 值
    """
    if preds.ndim == 4:
        preds = torch.argmax(preds, dim=1)

    preds = preds.to(device)
    labels = labels.to(device)

    # Mask for ignoring the specified index
    mask = torch.ones_like(labels, dtype=torch.bool, device=device)
    if ignore_index is not None:
        mask &= labels != ignore_index

    iou_per_class = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_inds = (preds == cls) & mask
        label_inds = (labels == cls) & mask

        # Intersection and union using GPU
        intersection = torch.sum(pred_inds & label_inds).item()
        union = torch.sum(pred_inds | label_inds).item()

        if union == 0:
            continue

        iou_per_class.append(intersection / union)

    if len(iou_per_class) == 0:
        return float('nan')
    
    return torch.mean(torch.tensor(iou_per_class, device=device)).item()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, C, H, W)   targets: (B, H, W)
        num_classes = logits.size(1)
        mask = targets != self.ignore_index
        targets = targets * mask
        logits  = logits * mask.unsqueeze(1)

        # one-hot
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)

        dims = (0,) + tuple(range(2, targets.ndimension() + 1))
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality  = torch.sum(probs + targets_onehot, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()


# -----------------------------------------------
# 示例：训练循环
# -----------------------------------------------
def train_one_epoch(model, data_loader, optimizer, criterion, device, scheduler,ce_loss,dice_loss):
    model.train()
    running_loss = 0.0
    running_total_accuracy = 0.0
    running_accuracy = torch.zeros(150).to(device)
    running_intersection = torch.zeros(150).to(device)
    running_union = torch.zeros(150).to(device)
    #tong_miou_total = 0.0
    #num_batches = 0
    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # 输出尺寸可能与 masks 不完全匹配，此处根据情况调整尺寸
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        loss = ce_loss(outputs, masks)+0.7*dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        scheduler.step() 
        running_loss += loss.item() * images.size(0)
        running_total_accuracy += calculate_accuracy(outputs, masks) * images.size(0)
        accuracy1, intersection, union = calculate_iou(outputs, masks, device)
        running_accuracy += accuracy1 * images.size(0)
        running_intersection += intersection
        running_union += union
        #tong_miou_total += mean_iou(outputs, masks, device, num_classes=150, ignore_index=0)
        #num_batches += 1
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = running_accuracy / len(data_loader.dataset)
    epoch_total_accuracy = running_total_accuracy / len(data_loader.dataset)
    running_iou = running_intersection / (running_union + 1e-6)
    #epoch_miou = tong_miou_total / num_batches if num_batches > 0 else 0.0
    print(len(data_loader.dataset))
    return epoch_loss, epoch_accuracy, epoch_total_accuracy, running_iou#, epoch_miou


def check_dataset_labels(data_loader, num_classes):
    for images, masks, image_names, mask_names in data_loader:
        for i in range(images.size(0)):
            unique_labels = torch.unique(masks[i])
            for label in unique_labels:
                if label < 0 or label >= num_classes:
                    print(f"Warning: Found label {label} which is out of bounds [0, {num_classes - 1}]")
                    print(f"Image: {image_names[i]}, Mask: {mask_names[i]}")
                    # 可以选择在这里进一步处理，例如记录这些标签或抛出异常


def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_total_accuracy = 0.0
    running_accuracy = torch.zeros(150).to(device)
    running_intersection = torch.zeros(150).to(device)
    running_union = torch.zeros(150).to(device)
    recal_per_class = torch.zeros(150).to(device)
    recal = torch.zeros(150).to(device)
    #tong_miou_total = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            # 输出尺寸可能与 masks 不完全匹配，此处根据情况调整尺寸
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)

            # 计算每个 batch 的损失和准确率
            running_loss += loss.item() * images.size(0)
            running_total_accuracy += calculate_accuracy(outputs, masks) * images.size(0)
            accuracy1, intersection, union = calculate_iou(outputs, masks, device)
            running_accuracy += accuracy1 * images.size(0)
            running_intersection += intersection
            running_union += union
            recal_per_class += calculate_recall(outputs, masks, device)
            #tong_miou_total += mean_iou(outputs, masks, device, num_classes=150, ignore_index=0)
            num_batches += 1
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = running_accuracy / len(data_loader.dataset)
    epoch_total_accuracy = running_total_accuracy / len(data_loader.dataset)
    running_iou = running_intersection / (running_union + 1e-6)
    recal = recal_per_class / num_batches
    #epoch_miou = tong_miou_total / num_batches if num_batches > 0 else 0.0
    return epoch_loss, epoch_accuracy, epoch_total_accuracy, running_iou, recal#, epoch_miou


# -----------------------------------------------
# 主函数：组装模型、数据及训练
# -----------------------------------------------
def main():
    # 训练循环
    num_epochs = 300
    best_val_accuracy = 0.0
    bestval_loss = float('inf')
    bestval_epoch = 0
    early_stopping_counter = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = 151  # ADE20K 一般有150个类别
    model = DeepLabV3Plus(num_classes=num_classes, backbone='resnet50', pretrained_backbone=True)
    model = model.to(device)
    log_filename = f"training_log_120.txt"

    # 定义损失函数（例如交叉熵损失）和优化器
        # 为重要类别设置更高的损失权重
    class_weights = torch.ones(151).to(device)  # ADE20K有151类
    important_classes = [1, 4, 6, 9, 15, 43]
    # 方法1：固定权重（简单有效）
    for cls in important_classes:
        class_weights[cls] = 5.0  # 重要类别权重设为3倍
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设 0 为 ignore 标签
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)   # 忽略 0 类
    dice_loss = DiceLoss(ignore_index=0)              # 用你的实现
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.003,momentum=0.9,weight_decay=1e-4)

    # 数据加载器
    train_loader = get_data_loaders(root_dir='ADE20/train', batch_size=16)
    val_loader = get_valdata_loaders(root_dir='ADE20/val', batch_size=8)
    max_iter   = num_epochs * len(train_loader)      # 总迭代步数
    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: (1 - step / max_iter) ** 0.3)
    # 检查数据集标签
    # check_dataset_labels(train_loader, num_classes)

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_total_accuracy, train_miou = train_one_epoch(model, train_loader, optimizer, criterion,
                                                                           device, scheduler,ce_loss,dice_loss)
        # 验证阶段
        val_loss, val_accuracy, val_total_accuracy, val_miou, recal = validate(model, val_loader, criterion, device)

        train_accuracy_mean = torch.mean(train_accuracy)
        val_accuracy_mean = torch.mean(val_accuracy)
        train_accuracy_miou = torch.mean(train_miou)
        val_accuracy_miou = torch.mean(val_miou)
        print(f"Epoch [{epoch + 1}/{num_epochs}], \n"
              f"Train Loss: {train_loss:.4f}, train total accuracy: {train_total_accuracy:.4f}, Train mean IOU: {train_accuracy_miou:.4f}, Train mean accuracy: {train_accuracy_mean:.4f} \n"
              f"Val Loss: {val_loss:.4f}, val total accuracy: {val_total_accuracy:.4f}, VAL mean IOU: {val_accuracy_miou:.4f}, Val mean accuracy: {val_accuracy_mean:.4f}")
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 打开文件并写入内容
        with open(log_filename, 'a') as f:  # 使用 'a' 模式以追加方式写入文件
            print(f"{current_time}: Epoch [{epoch + 1}/{num_epochs}], \n"
                  f"Train Loss: {train_loss:.4f}, train total accuracy: {train_total_accuracy:.4f}, Train mean IOU: {train_accuracy_miou:.4f}, Train mean accuracy: {train_accuracy_mean:.4f},\n"
                  f" Train Accuracy: {train_miou.tolist()}, \n"
                  f"Val Loss: {val_loss:.4f}, val total accuracy: {val_total_accuracy:.4f}, VAL mean IOU: {val_accuracy_miou:.4f}, Val mean accuracy: {val_accuracy_mean:.4f},\n"
                  f" Val Accuracy: {val_miou.tolist()}\n"
                  f" Val recall: {recal.tolist()}\n",
                  file=f)  # 将 print 的内容写入文件    
        # 每个 epoch 后保存模型（根据验证准确率判断是否保存最佳模型）
        if train_accuracy_miou > best_val_accuracy:
            best_val_accuracy = train_accuracy_miou
            torch.save(model.state_dict(), 'deeplabv3plus_best_120.pth')  # 保存当前模型为最佳模型
            print(f"Model saved at epoch {epoch + 1} with validation accuracy: {val_total_accuracy:.4f}")
            bestval_epoch = epoch
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 30:
                print(f"Early stopping at epoch {bestval_epoch}")
                break
    # 保存模型
    torch.save(model.state_dict(), 'deeplabv3plus_ade20k_120.pth')


if __name__ == '__main__':
    main()
