import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import os


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


# -----------------------------------------------
# 图像预处理函数（与训练时一致）
# -----------------------------------------------
def preprocess_image(image_path, target_size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加 batch 维度


# -----------------------------------------------
# 预测函数
# -----------------------------------------------
def predict(model, image_path, device='cuda'):
    # 1. 预处理图像
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # 2. 推理
    with torch.no_grad():
        output = model(input_tensor)
        if output.shape[-2:] != input_tensor.shape[-2:]:
            output = F.interpolate(output, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    return pred_mask


# -----------------------------------------------
# 可视化函数
# -----------------------------------------------
def visualize_prediction(image_path, pred_mask, save_path=None):
    # 1. 读取原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize((pred_mask.shape[1], pred_mask.shape[0]))

    # 2. 创建彩色掩码（假设你的数据集有150类 + 背景）
    palette = np.random.randint(0, 255, size=(151, 3))  # 与 ADE20K 的151类对应
    palette[0] = [0, 0, 0]  # 背景设为黑色
    colored_mask = palette[pred_mask]

    # 3. 叠加显示
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(colored_mask)
    plt.title('Predicted Mask')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# -----------------------------------------------
# 主函数
# -----------------------------------------------
def main():
    # 1. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3Plus(num_classes=151, backbone='resnet101', pretrained_backbone=False)

    # 2. 加载训练好的权重
    checkpoint_path = 'deeplabv3plus.pth'  # 替换为你的 .pth 文件路径
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 预测单张图像
    image_path = 'C:/Users/Admin/Desktop/图片1.jpg'  # 替换为你的测试图像路径
    pred_mask = predict(model, image_path, device)

    # 4. 可视化结果
    visualize_prediction(image_path, pred_mask, save_path='prediction_result.png')


if __name__ == '__main__':
    main()