import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import copy
import multiprocessing
import warnings
import glob
import torch.nn.functional as F

# 新增：导入transformers库和PEFT库用于CLIP模型和LoRA
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings('ignore')

# --- 1. 优化后的配置参数 ---
# 路径设置
BASE_DIR = "./lung_cancer_data"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "clip_lora_best_model.pth")

# 本地数据集路径
LOCAL_DATASET_PATH = "/Users/huangxh/Documents/DMECL/LC25000"

# 优化后的训练参数
BATCH_SIZE = 16  # 保持不变
NUM_EPOCHS = 15  # 增加训练轮数
LEARNING_RATE = 0.0005  # 降低学习率
PATIENCE = 8  # 增加耐心值

# CLIP模型配置
LOCAL_CLIP_MODEL_PATH = "/Users/huangxh/Documents/DMECL/clip-vit-base-patch32-local"

# 方案3：优化后的LoRA配置参数
LORA_CONFIG = {
    "r": 32,  # 增加LoRA rank
    "lora_alpha": 64,  # 增加alpha值
    "lora_dropout": 0.05,  # 降低dropout
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "out_proj",  # attention layers
        "fc1", "fc2"  # feed forward layers
    ]
}

# 设备检测
def get_device():
    """
    获取最佳可用设备，优先级：MPS > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# LC25000数据集的类别
CLASSES = ['lung_aca', 'lung_n', 'lung_scc']  # 肺腺癌、正常、肺鳞癌
NUM_CLASSES = len(CLASSES)

print(f"Using device: {DEVICE}")

# --- 方案2：损失函数优化 ---
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing交叉熵损失，有助于提高泛化能力
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss，帮助模型关注难分类样本
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

# --- 方案4：增强的CLIP + LoRA + 分类头模型定义 ---
class CLIPLoRAClassifier(nn.Module):
    """
    CLIP Vision Encoder + LoRA + 增强分类头模型
    """
    def __init__(self, clip_model_name=LOCAL_CLIP_MODEL_PATH, num_classes=NUM_CLASSES, 
                 lora_config=LORA_CONFIG):
        super(CLIPLoRAClassifier, self).__init__()
        
        # 加载基础CLIP模型
        print("Loading base CLIP model...")
        self.base_clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 冻结所有CLIP参数
        for param in self.base_clip_model.parameters():
            param.requires_grad = False
        print("✅ All CLIP parameters frozen")
        
        # 配置LoRA
        self.lora_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # 用于特征提取任务
        )
        
        # 应用LoRA到CLIP vision model
        print("Applying LoRA to CLIP vision model...")
        self.clip_model = get_peft_model(self.base_clip_model, self.lora_config)
        
        # 获取CLIP vision encoder的输出维度
        self.clip_hidden_size = self.base_clip_model.config.vision_config.hidden_size
        
        # 方案4：增强的分类头 - 更深层的网络
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.clip_hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # 打印模型信息
        self._print_model_info()
        
    def _print_model_info(self):
        """打印模型参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n=== Enhanced Model Information ===")
        print(f"CLIP hidden size: {self.clip_hidden_size}")
        print(f"Enhanced Classification head: {self.clip_hidden_size} -> 1024 -> 512 -> 256 -> {NUM_CLASSES}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # 打印LoRA配置
        print(f"\n=== Enhanced LoRA Configuration ===")
        print(f"LoRA rank (r): {self.lora_config.r}")
        print(f"LoRA alpha: {self.lora_config.lora_alpha}")
        print(f"LoRA dropout: {self.lora_config.lora_dropout}")
        print(f"Target modules: {self.lora_config.target_modules}")
        
        # 打印可训练参数详情
        print(f"\n=== Trainable Parameters ===")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} parameters")
    
    def forward(self, pixel_values):
        # 通过CLIP vision encoder (with LoRA) 提取特征
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        
        # 获取pooled输出 (CLS token的表示)
        pooled_output = vision_outputs.pooler_output  # [batch_size, hidden_size]
        
        # 通过增强的分类头
        logits = self.classifier(pooled_output)
        
        return logits
    
    def save_lora_weights(self, save_path):
        """保存LoRA权重"""
        self.clip_model.save_pretrained(save_path)
        print(f"LoRA weights saved to: {save_path}")
    
    def load_lora_weights(self, load_path):
        """加载LoRA权重"""
        from peft import PeftModel
        self.clip_model = PeftModel.from_pretrained(self.base_clip_model, load_path)
        print(f"LoRA weights loaded from: {load_path}")

# --- 3. 增强的图像预处理函数 ---
def get_clip_transforms():
    """
    针对医学图像的增强数据增强策略
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 医学图像可以垂直翻转
            transforms.RandomRotation(degrees=20),  # 增加旋转角度
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # 医学图像特定增强
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    }
    
    return data_transforms

# --- 4. 修改后的本地数据集类 ---
class LocalImageDataset(Dataset):
    """
    本地图像数据集类，适配CLIP输入格式
    """
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None, processor=None):
        self.original_image_paths = image_paths.copy()
        self.original_labels = labels.copy()
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        self.processor = processor
        
        # 验证并过滤有效的图像
        self._validate_images()
        
    def _validate_images(self):
        """验证图像文件并移除损坏的文件"""
        print("Validating image files...")
        valid_indices = []
        invalid_count = 0
        
        for idx, image_path in enumerate(tqdm(self.original_image_paths, desc="Validating images")):
            try:
                if not os.path.exists(image_path):
                    invalid_count += 1
                    continue
                    
                file_size = os.path.getsize(image_path)
                if file_size <= 200:
                    invalid_count += 1
                    continue
                
                try:
                    with Image.open(image_path) as img:
                        _ = img.size
                        _ = img.mode
                    valid_indices.append(idx)
                except Exception:
                    invalid_count += 1
                        
            except Exception as e:
                invalid_count += 1
        
        # 更新有效的图像路径和标签
        self.image_paths = [self.original_image_paths[i] for i in valid_indices]
        self.labels = [self.original_labels[i] for i in valid_indices]
        
        print(f"Validation complete: {len(self.image_paths)} valid images, {invalid_count} invalid images removed")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 获取标签
        label = self.labels[idx]
        
        # 应用变换
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for {image_path}: {e}")
                image = torch.zeros(3, 224, 224)
        
        if self.processor:
            try:
                # CLIPProcessor会将PIL图像转换为tensor并进行标准化
                processed = self.processor(images=image, return_tensors="pt")
                image = processed['pixel_values'].squeeze(0)  # 移除batch维度
            except Exception as e:
                print(f"CLIPProcessor failed for {image_path}: {e}")
                image = torch.zeros(3, 224, 224)
        else:
            # 如果没有processor，使用默认的tensor转换
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)

        # 处理标签
        if isinstance(label, str) and self.class_to_idx:
            label = self.class_to_idx.get(label, 0)
        
        return image, label

# --- 5. 数据加载函数（保持基本不变） ---
def load_local_dataset():
    """从本地目录加载LC25000数据集"""
    print("--- Loading local LC25000 dataset ---")
    
    if not os.path.exists(LOCAL_DATASET_PATH):
        raise FileNotFoundError(f"Dataset directory not found: {LOCAL_DATASET_PATH}")
    
    image_paths = []
    labels = []
    class_to_idx = {}
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    class_dirs = [d for d in os.listdir(LOCAL_DATASET_PATH) 
                  if os.path.isdir(os.path.join(LOCAL_DATASET_PATH, d)) and not d.startswith('.')]
    
    if class_dirs:
        print("Found class directories:", class_dirs)
        
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_dirs))}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
        for class_name in class_dirs:
            class_dir = os.path.join(LOCAL_DATASET_PATH, class_name)
            
            class_images = []
            for ext in image_extensions:
                class_images.extend(glob.glob(os.path.join(class_dir, ext)))
            
            print(f"Found {len(class_images)} potential images in class '{class_name}'")
            
            valid_images = []
            for img_path in class_images:
                if os.path.getsize(img_path) > 200:
                    valid_images.append(img_path)
            
            print(f"Valid images in class '{class_name}': {len(valid_images)}")
            
            image_paths.extend(valid_images)
            labels.extend([class_name] * len(valid_images))
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    
    return image_paths, labels, class_to_idx, idx_to_class

def create_data_splits_local(image_paths, labels, class_to_idx):
    """将本地数据集分割为训练、验证和测试集"""
    print("--- Creating data splits ---")
    
    indices = list(range(len(image_paths)))
    
    try:
        numeric_labels = [class_to_idx[label] if isinstance(label, str) else label 
                         for label in labels]
        
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=numeric_labels
        )
        
        temp_labels = [numeric_labels[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
    except Exception as e:
        print(f"Stratified split failed: {e}, using random split")
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42
        )
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    splits = {}
    for split_name, indices in [('train', train_indices), 
                               ('validation', val_indices), 
                               ('test', test_indices)]:
        splits[split_name] = {
            'image_paths': [image_paths[i] for i in indices],
            'labels': [labels[i] for i in indices]
        }
    
    return splits

def create_data_loaders_local():
    """创建本地数据集的数据加载器，使用CLIP预处理"""
    print("--- Creating data loaders ---")
    
    image_paths, labels, class_to_idx, idx_to_class = load_local_dataset()
    
    if len(image_paths) == 0:
        print("No valid images found. Please download Git LFS files first.")
        return None, None, None, None, None, None
    
    data_splits = create_data_splits_local(image_paths, labels, class_to_idx)
    
    processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_MODEL_PATH)
    data_transforms = get_clip_transforms()
    
    pytorch_datasets = {}
    for split in ['train', 'validation', 'test']:
        pytorch_datasets[split] = LocalImageDataset(
            data_splits[split]['image_paths'],
            data_splits[split]['labels'],
            data_transforms.get(split, data_transforms['validation']),
            class_to_idx,
            processor
        )
    
    dataloaders = {}
    for split in ['train', 'validation', 'test']:
        dataloaders[split] = DataLoader(
            pytorch_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    dataset_sizes = {split: len(pytorch_datasets[split]) for split in ['train', 'validation', 'test']}
    
    return pytorch_datasets, dataloaders, dataset_sizes, class_to_idx, idx_to_class, data_splits

# --- 6. 模型相关函数 ---
def load_clip_lora_model():
    """加载CLIP + LoRA + 分类头模型"""
    print(f"Loading enhanced CLIP + LoRA model: {LOCAL_CLIP_MODEL_PATH}")
    model = CLIPLoRAClassifier(
        clip_model_name=LOCAL_CLIP_MODEL_PATH,
        num_classes=NUM_CLASSES,
        lora_config=LORA_CONFIG
    )
    return model

# --- 7. 测试函数 ---
def test_model(model, test_dataloader, dataset_size, class_to_idx, idx_to_class):
    """
    测试训练好的CLIP + LoRA模型
    """
    print("\n--- Testing Enhanced CLIP + LoRA model on test dataset ---")
    
    model.eval()
    
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # 用于计算每个类别的准确率
    class_correct = {i: 0 for i in range(len(class_to_idx))}
    class_total = {i: 0 for i in range(len(class_to_idx))}
    
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Testing Enhanced CLIP + LoRA")
        
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 统计正确预测
            running_corrects += torch.sum(preds == labels.data)
            
            # 保存预测结果用于详细分析
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 统计每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    # 计算总体准确率
    test_acc = running_corrects.item() / dataset_size
    
    print(f"\n=== Enhanced CLIP + LoRA Test Results ===")
    print(f"Overall Test Accuracy: {test_acc:.4f} ({running_corrects}/{dataset_size})")
    
    # 打印每个类别的准确率
    print(f"\nPer-class Accuracy:")
    for class_idx, class_name in idx_to_class.items():
        if class_total[class_idx] > 0:
            class_acc = class_correct[class_idx] / class_total[class_idx]
            print(f"  {class_name}: {class_acc:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")
        else:
            print(f"  {class_name}: No samples in test set")
    
    return test_acc, all_preds, all_labels

# --- 8. 优化后的训练函数 ---
def train_model():
    """使用优化后的CLIP + LoRA模型训练本地LC25000数据集"""
    print("\n--- Training with Enhanced CLIP + LoRA ---")
    print(f"Using device: {DEVICE}")
    print(f"CLIP model: {LOCAL_CLIP_MODEL_PATH}")

    # 创建数据加载器
    result = create_data_loaders_local()
    if result[0] is None:
        print("Error: Failed to create data loaders")
        return None
    
    pytorch_datasets, dataloaders, dataset_sizes, class_to_idx, idx_to_class, data_splits = result
    
    if dataset_sizes['train'] == 0:
        print("Error: No training data found.")
        return None

    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {list(class_to_idx.keys())}")

    # 加载增强的CLIP + LoRA模型
    model = load_clip_lora_model()
    model = model.to(DEVICE)

    # 方案2：组合损失函数
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ls = LabelSmoothingCrossEntropy(smoothing=0.1)
    criterion_focal = FocalLoss(alpha=1, gamma=2)
    
    def combined_loss(outputs, labels):
        loss_ce = criterion_ce(outputs, labels)
        loss_ls = criterion_ls(outputs, labels)
        loss_focal = criterion_focal(outputs, labels)
        return 0.4 * loss_ce + 0.3 * loss_ls + 0.3 * loss_focal

    # 方案3：分组参数，使用不同学习率
    lora_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora' in name.lower():
                lora_params.append(param)
            else:
                classifier_params.append(param)
    
    # 分类头使用更高学习率，LoRA使用较低学习率
    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': LEARNING_RATE * 0.5},  # LoRA较低学习率
        {'params': classifier_params, 'lr': LEARNING_RATE * 2}  # 分类头更高学习率
    ], weight_decay=1e-4)
    
    print("✅ Using differentiated learning rates:")
    print(f"  LoRA parameters: {LEARNING_RATE * 0.5}")
    print(f"  Classifier parameters: {LEARNING_RATE * 2}")
    
    # 使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # 训练循环
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    
    # 添加梯度裁剪
    max_grad_norm = 1.0

    print("Starting enhanced training...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # 详细的类别统计
            class_corrects = {i: 0 for i in range(NUM_CLASSES)}
            class_totals = {i: 0 for i in range(NUM_CLASSES)}

            pbar = tqdm(dataloaders[phase], desc=f"Epoch {epoch+1} - {phase}")
            
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    try:
                        outputs = model(inputs)
                        loss = combined_loss(outputs, labels)  # 使用组合损失
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    except RuntimeError as e:
                        print(f"Runtime error during {phase}: {e}")
                        continue

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 统计每个类别的准确率
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    class_totals[label] += 1
                    if label == pred:
                        class_corrects[label] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 打印每个类别的准确率
            print(f'{phase} Per-class Accuracy:')
            for class_idx, class_name in idx_to_class.items():
                if class_totals[class_idx] > 0:
                    class_acc = class_corrects[class_idx] / class_totals[class_idx]
                    print(f'  {class_name}: {class_acc:.4f} ({class_corrects[class_idx]}/{class_totals[class_idx]})')

            if phase == 'validation':
                # 计算平均类别准确率（更关注少数类别）
                valid_class_accs = []
                for class_idx in range(NUM_CLASSES):
                    if class_totals[class_idx] > 0:
                        class_acc = class_corrects[class_idx] / class_totals[class_idx]
                        valid_class_accs.append(class_acc)
                
                avg_class_acc = sum(valid_class_accs) / len(valid_class_accs) if valid_class_accs else 0
                
                # 使用平均类别准确率作为保存标准
                if avg_class_acc > best_acc:
                    best_acc = avg_class_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # 保存完整模型
                    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    
                    # 单独保存LoRA权重
                    lora_save_path = os.path.join(BASE_DIR, "lora_weights")
                    model.save_lora_weights(lora_save_path)
                    
                    print(f"New best model saved with avg class accuracy: {best_acc:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

        if phase == 'train':
            scheduler.step()  # 更新学习率

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f'Enhanced training completed in {total_time:.0f}s')
    print(f'Best average class accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, class_to_idx, idx_to_class, dataloaders

# --- 9. 主函数 ---
def main():
    """主函数"""
    try:
        print("=== Enhanced CLIP + LoRA LC25000 Lung Cancer Classification ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {DEVICE}")
        print(f"CLIP model: {LOCAL_CLIP_MODEL_PATH}")
        print(f"Local dataset path: {LOCAL_DATASET_PATH}")
        print()

        # 检查必要的库
        try:
            import transformers
            import peft
            print(f"✓ Transformers version: {transformers.__version__}")
            print(f"✓ PEFT version: {peft.__version__}")
        except ImportError as e:
            print(f"❌ Missing library: {e}")
            print("Please install: pip install transformers peft")
            return

        # 检查本地数据集路径
        if not os.path.exists(LOCAL_DATASET_PATH):
            print(f"✗ Dataset directory not found: {LOCAL_DATASET_PATH}")
            return
        else:
            print(f"✓ Dataset directory found: {LOCAL_DATASET_PATH}")

        # 训练模型
        result = train_model()
        
        if result is not None:
            model, class_to_idx, idx_to_class, dataloaders = result
            print("Enhanced training completed successfully!")
            print(f"Model saved at: {MODEL_SAVE_PATH}")
            print(f"LoRA weights saved at: {os.path.join(BASE_DIR, 'lora_weights')}")
            print(f"Class mapping: {class_to_idx}")

            # 在测试集上评估模型
            if 'test' in dataloaders:
                # 重新创建数据加载器以获取dataset_sizes
                _, _, dataset_sizes, _, _, _ = create_data_loaders_local()
                
                test_acc, test_preds, test_labels = test_model(
                    model, 
                    dataloaders['test'], 
                    dataset_sizes['test'], 
                    class_to_idx, 
                    idx_to_class
                )
                
                print(f"\n🎉 Final Enhanced CLIP + LoRA Test Accuracy: {test_acc:.4f}")
            else:
                print("❌ Test dataloader not available")
        else:
            print("Training failed!")
            return
        
        print("\n=== Enhanced Training completed! ===")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()