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

# 新增：导入transformers库用于CLIP模型
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings('ignore')

# --- 1. 配置参数 ---
# 路径设置
BASE_DIR = "./lung_cancer_data"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "clip_best_model.pth")

# 本地数据集路径
LOCAL_DATASET_PATH = "/Users/huangxh/Documents/DMECL/LC25000"

# 训练参数
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# CLIP模型配置
# 本地CLIP模型路径（修改这里）
LOCAL_CLIP_MODEL_PATH = "/Users/huangxh/Documents/DMECL/clip-vit-base-patch32-local"
# CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 可选择其他CLIP模型

# 设备检测 - 支
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
if DEVICE.type == "mps":
    print("✅ Apple Silicon GPU (MPS) acceleration enabled!")
    print("💡 Tip: You can try increasing BATCH_SIZE for better performance")
elif DEVICE.type == "cuda":
    print("✅ NVIDIA GPU (CUDA) acceleration enabled!")
else:
    print("⚠️  Using CPU - consider upgrading PyTorch for MPS support")

print(f"Available CPU cores: {multiprocessing.cpu_count()}")

# --- 2. CLIP + 分类头模型定义 ---
class CLIPClassifier(nn.Module):
    """
    CLIP Vision Encoder + 分类头模型
    """
    def __init__(self, clip_model_name=LOCAL_CLIP_MODEL_PATH, num_classes=NUM_CLASSES, freeze_clip=False):
        super(CLIPClassifier, self).__init__()
        
        # 加载CLIP vision encoder
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 是否冻结CLIP参数
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("CLIP vision encoder parameters frozen")
        else:
            print("CLIP vision encoder parameters will be fine-tuned")
        
        # 获取CLIP vision encoder的输出维度
        self.clip_hidden_size = self.clip_model.config.vision_config.hidden_size
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.clip_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        print(f"CLIP hidden size: {self.clip_hidden_size}")
        print(f"Classification head: {self.clip_hidden_size} -> 512 -> {num_classes}")
    
    def forward(self, pixel_values):
        # 通过CLIP vision encoder提取特征
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        
        # 获取pooled输出 (CLS token的表示)
        pooled_output = vision_outputs.pooler_output  # [batch_size, hidden_size]
        
        # 通过分类头
        logits = self.classifier(pooled_output)
        
        return logits

# --- 3. 修改图像预处理函数 ---
def get_clip_transforms():
    """
    获取CLIP模型的图像预处理变换
    CLIP通常使用224x224的输入尺寸
    """
    
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),  # CLIP使用224x224
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10)
            
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)  # CLIP使用224x224
            
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    }
    
    return data_transforms

# --- 4. 修改后的图像验证函数 ---
def is_valid_image(image_path):
    """
    检查图像文件是否有效 - 修复版本
    """
    try:
        # 检查文件大小
        if os.path.getsize(image_path) <= 200:  # Git LFS指针文件通常很小
            return False
        
        # 尝试打开并读取图像
        with Image.open(image_path) as img:
            # 获取图像格式和尺寸
            img.load()  # 确保图像数据被加载
            _ = img.size  # 获取尺寸
            _ = img.mode  # 获取模式
        return True
    except Exception as e:
        print(f"Invalid image {image_path}: {e}")
        return False

# --- 5. 修改后的本地数据集类 ---
class LocalImageDataset(Dataset):
    """
    本地图像数据集类，适配CLIP输入格式
    """
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None,processor=None):
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
                # 基本检查：文件存在且不为空
                if not os.path.exists(image_path):
                    invalid_count += 1
                    continue
                    
                # 检查是否是Git LFS指针文件
                file_size = os.path.getsize(image_path)
                if file_size <= 200:
                    invalid_count += 1
                    continue
                
                # 尝试快速打开图像
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
            # 创建一个默认图像
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 获取标签
        label = self.labels[idx]
        
        # 应用变换
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for {image_path}: {e}")
                # 如果变换失败，创建一个默认的tensor
                image = torch.zeros(3, 224, 224)  # CLIP使用224x224
        
        if self.processor:
            try:
                # CLIPProcessor会将PIL图像转换为tensor并进行标准化
                processed = self.processor(images=image, return_tensors="pt")
                image = processed['pixel_values'].squeeze(0)  # 移除batch维度
            except Exception as e:
                print(f"CLIPProcessor failed for {image_path}: {e}")
                # 如果处理失败，创建一个默认的tensor
                image = torch.zeros(3, 224, 224)
        else:
            # 如果没有processor，使用默认的tensor转换
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)

        # 处理标签
        if isinstance(label, str) and self.class_to_idx:
            label = self.class_to_idx.get(label, 0)
        
        return image, label

# --- 6. 数据加载函数（保持不变） ---
def load_local_dataset():
    """
    从本地目录加载LC25000数据集，带有文件验证
    """
    print("--- Loading local LC25000 dataset ---")
    
    if not os.path.exists(LOCAL_DATASET_PATH):
        raise FileNotFoundError(f"Dataset directory not found: {LOCAL_DATASET_PATH}")
    
    image_paths = []
    labels = []
    class_to_idx = {}
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 方法1: 如果数据按类别分在不同文件夹中
    class_dirs = [d for d in os.listdir(LOCAL_DATASET_PATH) 
                  if os.path.isdir(os.path.join(LOCAL_DATASET_PATH, d)) and not d.startswith('.')]
    
    if class_dirs:
        print("Found class directories:", class_dirs)
        
        # 创建类别映射
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_dirs))}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
        # 加载每个类别的图像
        for class_name in class_dirs:
            class_dir = os.path.join(LOCAL_DATASET_PATH, class_name)
            
            # 获取该类别的所有图像文件
            class_images = []
            for ext in image_extensions:
                class_images.extend(glob.glob(os.path.join(class_dir, ext)))
            
            print(f"Found {len(class_images)} potential images in class '{class_name}'")
            
            # 验证图像文件
            valid_images = []
            for img_path in class_images:
                if os.path.getsize(img_path) > 200:  # 过滤Git LFS指针文件
                    valid_images.append(img_path)
            
            print(f"Valid images in class '{class_name}': {len(valid_images)}")
            
            # 添加到总列表
            image_paths.extend(valid_images)
            labels.extend([class_name] * len(valid_images))
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    
    return image_paths, labels, class_to_idx, idx_to_class

def create_data_splits_local(image_paths, labels, class_to_idx):
    """
    将本地数据集分割为训练、验证和测试集
    """
    print("--- Creating data splits ---")
    
    # 获取所有索引
    indices = list(range(len(image_paths)))
    
    # 分层采样
    try:
        # 将标签转换为数字
        numeric_labels = [class_to_idx[label] if isinstance(label, str) else label 
                         for label in labels]
        
        # 70% 训练, 30% 临时
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=numeric_labels
        )
        
        # 从临时数据中分出验证和测试 (15% + 15%)
        temp_labels = [numeric_labels[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
    except Exception as e:
        print(f"Stratified split failed: {e}, using random split")
        # 随机分割
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42
        )
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # 创建分割后的数据
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
    """
    创建本地数据集的数据加载器，使用CLIP预处理
    """
    print("--- Creating data loaders ---")
    
    # 加载本地数据集
    image_paths, labels, class_to_idx, idx_to_class = load_local_dataset()
    
    if len(image_paths) == 0:
        print("No valid images found. Please download Git LFS files first.")
        return None, None, None, None, None, None
    
    # 分割数据
    data_splits = create_data_splits_local(image_paths, labels, class_to_idx)
    

    processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_MODEL_PATH)
    # 使用CLIP的数据预处理
    data_transforms = get_clip_transforms()
    
    # 创建PyTorch Dataset
    pytorch_datasets = {}
    for split in ['train', 'validation', 'test']:
        pytorch_datasets[split] = LocalImageDataset(
            data_splits[split]['image_paths'],
            data_splits[split]['labels'],
            data_transforms.get(split, data_transforms['validation']),
            class_to_idx,
            processor
        )
    
    # 创建数据加载器
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

# --- 7. 模型相关函数 ---
def load_clip_model(freeze_clip=False):
    """
    加载CLIP + 分类头模型
    """
    print(f"Loading CLIP model: {LOCAL_CLIP_MODEL_PATH}")
    model = CLIPClassifier(
        clip_model_name=LOCAL_CLIP_MODEL_PATH,
        num_classes=NUM_CLASSES,
        freeze_clip=freeze_clip
    )
    return model

def test_model(model, test_dataloader, dataset_size, class_to_idx, idx_to_class):
    """
    测试训练好的模型
    """
    print("\n--- Testing model on test dataset ---")
    
    model.eval()
    
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # 用于计算每个类别的准确率
    class_correct = {i: 0 for i in range(len(class_to_idx))}
    class_total = {i: 0 for i in range(len(class_to_idx))}
    
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Testing")
        
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
    
    print(f"\n=== Test Results ===")
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

# --- 8. 训练函数 ---
def train_model(freeze_clip=False):
    """
    使用CLIP模型训练本地LC25000数据集
    """
    print("\n--- Training with CLIP Vision Encoder ---")
    print(f"Using device: {DEVICE}")
    print(f"CLIP model: {LOCAL_CLIP_MODEL_PATH}")
    print(f"Freeze CLIP: {freeze_clip}")

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

    # 加载CLIP模型
    model = load_clip_model(freeze_clip=freeze_clip)
    model = model.to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 如果冻结CLIP，只优化分类头
    if freeze_clip:
        optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        print("Only optimizing classification head parameters")
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        print("Optimizing all model parameters")
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练循环
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
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

            pbar = tqdm(dataloaders[phase], desc=f"Epoch {epoch+1} - {phase}")
            
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    try:
                        # CLIP模型没有辅助输出，直接前向传播
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    except RuntimeError as e:
                        print(f"Runtime error during {phase}: {e}")
                        continue

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"New best model saved with accuracy: {best_acc:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

        if phase == 'train':
            scheduler.step()

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, class_to_idx, idx_to_class, dataloaders

# --- 9. 诊断函数（保持不变） ---
def diagnose_dataset(dataset_path):
    """
    诊断数据集问题
    """
    print(f"\n=== Dataset Diagnosis for {dataset_path} ===")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory does not exist: {dataset_path}")
        return False
    
    # 检查目录结构
    items = os.listdir(dataset_path)
    print(f"Items in dataset directory: {items}")
    
    # 检查类别目录
    class_dirs = [d for d in items if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
    print(f"Class directories found: {class_dirs}")
    
    if not class_dirs:
        print("❌ No class directories found")
        return False
    
    # 检查每个类别的文件
    total_files = 0
    git_lfs_files = 0
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, class_dir)
        files = os.listdir(class_path)
        print(f"\nClass '{class_dir}':")
        print(f"  Total files: {len(files)}")
        
        # 检查文件类型
        image_files = []
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_files.append(f)
        
        print(f"  Image files: {len(image_files)}")
        
        # 检查前几个文件
        for i, img_file in enumerate(image_files[:3]):
            img_path = os.path.join(class_path, img_file)
            size = os.path.getsize(img_path)
            print(f"    Sample {i+1}: {img_file} ({size} bytes)")
            
            if size <= 200:
                git_lfs_files += 1
                print(f"      ⚠️  Git LFS pointer file (not downloaded)")
            else:
                # 尝试打开图像
                try:
                    with Image.open(img_path) as img:
                        print(f"      ✅ Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
                        total_files += 1
                except Exception as e:
                    print(f"      ❌ Cannot open: {e}")
    
    print(f"\nSummary:")
    print(f"  Valid image files: {total_files}")
    print(f"  Git LFS pointer files: {git_lfs_files}")
    
    if git_lfs_files > 0:
        print(f"\n⚠️  Found {git_lfs_files} Git LFS pointer files.")
        print("Please run the following commands to download the actual files:")
        print(f"  cd {dataset_path}")
        print("  git lfs pull")
    
    return total_files > 0

# --- 10. 主函数 ---
def main(freeze_clip=False):
    """
    主函数
    Args:
        freeze_clip: 是否冻结CLIP参数，只训练分类头
    """
    try:
        print("=== CLIP-based LC25000 Lung Cancer Classification ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {DEVICE}")
        print(f"CLIP model: {LOCAL_CLIP_MODEL_PATH}")
        print(f"Local dataset path: {LOCAL_DATASET_PATH}")
        print(f"Freeze CLIP: {freeze_clip}")
        print()

        # 检查transformers库
        try:
            import transformers
            print(f"✓ Transformers version: {transformers.__version__}")
        except ImportError:
            print("❌ Transformers library not found. Please install: pip install transformers")
            return

        # 检查本地数据集路径
        if not os.path.exists(LOCAL_DATASET_PATH):
            print(f"✗ Dataset directory not found: {LOCAL_DATASET_PATH}")
            print("Please make sure the LC25000 dataset is in the correct location.")
            return
        else:
            print(f"✓ Dataset directory found: {LOCAL_DATASET_PATH}")

        # 诊断数据集
        if not diagnose_dataset(LOCAL_DATASET_PATH):
            print("❌ Dataset diagnosis failed. Please check your dataset structure.")
            return

        # 训练模型
        result = train_model(freeze_clip=freeze_clip)
        
        if result is not None:
            model, class_to_idx, idx_to_class, dataloaders = result
            print("Training completed successfully!")
            print(f"Model saved at: {MODEL_SAVE_PATH}")
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
                
                print(f"\n🎉 Final Test Accuracy: {test_acc:.4f}")
            else:
                print("❌ Test dataloader not available")
        else:
            print("Training failed!")
            return
        
        print("\n=== Training completed! ===")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 可以选择是否冻结CLIP参数
    # freeze_clip=True: 只训练分类头，训练更快但可能精度稍低
    # freeze_clip=False: 微调整个模型，训练较慢但可能精度更高
    main(freeze_clip=True)  # 可以改为True来只训练分类头