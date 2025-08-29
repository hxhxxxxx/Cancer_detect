import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import copy
import multiprocessing
import warnings
import glob
warnings.filterwarnings('ignore')

# --- 1. 配置参数 ---
# 路径设置
BASE_DIR = "./lung_cancer_data"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "829_best_model.pth")

# 本地数据集路径
LOCAL_DATASET_PATH = "/Users/huangxh/Documents/DMECL/LC25000"

# 训练参数
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# 设备检测 - 支持Apple Silicon
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



# --- 2. 修改后的图像验证函数 ---
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

def load_and_verify_image(image_path):
    """
    安全地加载图像，处理损坏的文件 - 修复版本
    """
    try:
        image = Image.open(image_path).convert('RGB')
        # 尝试获取图像尺寸来确保图像完整
        _ = image.size
        # 尝试加载图像数据
        image.load()
        return image
    except Exception as e:
        print(f"Warning: Cannot load image {image_path}: {e}")
        return None

# --- 3. 修改后的本地数据集类 ---
class LocalImageDataset(Dataset):
    """
    本地图像数据集类，带有改进的错误处理
    """
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        self.original_image_paths = image_paths.copy()
        self.original_labels = labels.copy()
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        
        # 验证并过滤有效的图像 - 使用更宽松的验证
        self._validate_images()
        
    def _validate_images(self):
        """验证图像文件并移除损坏的文件 - 改进版本"""
        print("Validating image files...")
        valid_indices = []
        invalid_count = 0
        
        # 使用更宽松的验证策略
        for idx, image_path in enumerate(tqdm(self.original_image_paths, desc="Validating images")):
            try:
                # 基本检查：文件存在且不为空
                if not os.path.exists(image_path):
                    invalid_count += 1
                    print(f"File not found: {image_path}")
                    continue
                    
                # 检查是否是Git LFS指针文件
                file_size = os.path.getsize(image_path)
                if file_size <= 200:  # Git LFS指针文件通常很小
                    invalid_count += 1
                    print(f"Git LFS pointer file (not downloaded): {image_path}")
                    continue
                
                # 尝试快速打开图像（不使用verify）
                try:
                    with Image.open(image_path) as img:
                        # 只检查基本属性
                        _ = img.size
                        _ = img.mode
                    valid_indices.append(idx)
                except Exception as img_error:
                    # 如果PIL无法打开，尝试其他方法
                    try:
                        # 检查文件扩展名
                        ext = os.path.splitext(image_path)[1].lower()
                        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                            # 对于常见格式，给一次机会
                            valid_indices.append(idx)
                        else:
                            invalid_count += 1
                            print(f"Unsupported format: {image_path}")
                    except:
                        invalid_count += 1
                        print(f"Cannot process: {image_path}")
                        
            except Exception as e:
                invalid_count += 1
                print(f"Error validating {image_path}: {e}")
        
        # 更新有效的图像路径和标签
        self.image_paths = [self.original_image_paths[i] for i in valid_indices]
        self.labels = [self.original_labels[i] for i in valid_indices]
        
        print(f"Validation complete: {len(self.image_paths)} valid images, {invalid_count} invalid images removed")
        
        # 如果没有有效图像，给出详细信息
        if len(self.image_paths) == 0:
            print("WARNING: No valid images found!")
            print("This is likely because Git LFS files haven't been downloaded.")
            print("Please run: cd /Users/huangxh/Documents/DMECL/LC25000 && git lfs pull")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        
        # 尝试多种方式加载图像
        image = None
        try:
            image = Image.open(image_path).convert('RGB')
            # 确保图像可以被处理
            _ = image.size
        except Exception as e:
            print(f"Warning: Cannot load image {image_path}: {e}")
            # 创建一个默认图像
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))  # 灰色默认图像
        
        # 获取标签
        label = self.labels[idx]
        
        # 应用变换
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for {image_path}: {e}")
                # 如果变换失败，创建一个默认的tensor
                image = torch.zeros(3, 299, 299)
        
        # 处理标签
        if isinstance(label, str) and self.class_to_idx:
            label = self.class_to_idx.get(label, 0)
        
        return image, label

# --- 4. 数据加载函数 ---
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
                else:
                    print(f"Skipping Git LFS pointer file: {img_path}")
            
            print(f"Valid images in class '{class_name}': {len(valid_images)}")
            
            # 添加到总列表
            image_paths.extend(valid_images)
            labels.extend([class_name] * len(valid_images))
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    print(f"Class distribution:")
    for cls in class_to_idx.keys():
        count = labels.count(cls)
        print(f"  {cls}: {count} images")
    
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

# def create_data_loaders_local():
#     """
#     创建本地数据集的数据加载器
#     """
#     print("--- Creating data loaders ---")
    
#     # 加载本地数据集
#     image_paths, labels, class_to_idx, idx_to_class = load_local_dataset()
    
#     if len(image_paths) == 0:
#         print("No valid images found. Please download Git LFS files first.")
#         return None, None, None, None, None
    
#     # 分割数据
#     data_splits = create_data_splits_local(image_paths, labels, class_to_idx)
    
#     # 数据预处理和增强
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.Resize((320, 320)),  # 先resize到稍大的尺寸
#             transforms.RandomResizedCrop(299),  # Inception v3需要299x299
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             transforms.RandomRotation(10),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'validation': transforms.Compose([
#             transforms.Resize(320),
#             transforms.CenterCrop(299),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
    
#     # 创建PyTorch Dataset
#     pytorch_datasets = {}
#     for split in ['train', 'validation', 'test']:
#         pytorch_datasets[split] = LocalImageDataset(
#             data_splits[split]['image_paths'],
#             data_splits[split]['labels'],
#             data_transforms.get(split, data_transforms['validation']),
#             class_to_idx
#         )
    
#     # 修改num_workers设置，减少并发以避免问题
#     num_workers = 0  # 设置为0以避免多进程问题
    
#     # 创建数据加载器
#     dataloaders = {}
#     for split in ['train', 'validation']:
#         dataloaders[split] = DataLoader(
#             pytorch_datasets[split],
#             batch_size=BATCH_SIZE,
#             shuffle=(split == 'train'),
#             num_workers=num_workers,  # 使用单进程
#             pin_memory=torch.cuda.is_available(),
#             drop_last=True
#         )
    
#     dataset_sizes = {split: len(pytorch_datasets[split]) for split in ['train', 'validation']}
    
#     return pytorch_datasets, dataloaders, dataset_sizes, class_to_idx, idx_to_class

def create_data_loaders_local():
    """
    创建本地数据集的数据加载器
    """
    print("--- Creating data loaders ---")
    
    # 加载本地数据集
    image_paths, labels, class_to_idx, idx_to_class = load_local_dataset()
    
    if len(image_paths) == 0:
        print("No valid images found. Please download Git LFS files first.")
        return None, None, None, None, None, None  # 增加一个返回值
    
    # 分割数据
    data_splits = create_data_splits_local(image_paths, labels, class_to_idx)
    
    # 数据预处理和增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([  # 添加test的变换
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # 创建PyTorch Dataset
    pytorch_datasets = {}
    for split in ['train', 'validation', 'test']:
        pytorch_datasets[split] = LocalImageDataset(
            data_splits[split]['image_paths'],
            data_splits[split]['labels'],
            data_transforms.get(split, data_transforms['validation']),
            class_to_idx
        )
    
    # 修改num_workers设置，减少并发以避免问题
    num_workers = 0
    
    # 创建数据加载器 - 包括test
    dataloaders = {}
    for split in ['train', 'validation', 'test']:  # 添加test
        dataloaders[split] = DataLoader(
            pytorch_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False  # test时不要drop_last
        )
    
    dataset_sizes = {split: len(pytorch_datasets[split]) for split in ['train', 'validation', 'test']}
    
    return pytorch_datasets, dataloaders, dataset_sizes, class_to_idx, idx_to_class, data_splits  # 返回data_splits用于调试
# --- 5. 模型相关函数 ---
def load_inception_model():
    """加载Inception v3模型，兼容不同版本的torchvision"""
    try:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        print("Loaded model with new torchvision API")
    except AttributeError:
        model = models.inception_v3(pretrained=True)
        print("Loaded model with legacy torchvision API")
    return model

def test_model(model, test_dataloader, dataset_size, class_to_idx, idx_to_class):
    """
    测试训练好的模型
    """
    print("\n--- Testing model on test dataset ---")
    
    model.eval()  # 设置为评估模式
    
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
# --- 6. 训练函数 ---
def train_model():
    """
    使用Inception v3模型训练本地LC25000数据集
    """
    print("\n--- Training on local LC25000 dataset ---")
    print(f"Using device: {DEVICE}")

    # 创建数据加载器
    result = create_data_loaders_local()
    if result[0] is None:
        print("Error: Failed to create data loaders")
        return None
    
    pytorch_datasets, dataloaders, dataset_sizes, class_to_idx, idx_to_class,data_splits = result
    
    if dataset_sizes['train'] == 0:
        print("Error: No training data found.")
        return None

    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {list(class_to_idx.keys())}")

    # 加载预训练的Inception v3模型
    model = load_inception_model()
    
    # 替换分类器（根据实际类别数）
    num_ftrs = model.fc.in_features
    actual_num_classes = len(class_to_idx)
    model.fc = nn.Linear(num_ftrs, actual_num_classes)
    
    # 替换辅助分类器
    if hasattr(model, 'AuxLogits'):
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, actual_num_classes)

    model = model.to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
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
                        if phase == 'train' and hasattr(model, 'AuxLogits'):
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
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

# --- 7. 添加数据集诊断函数 ---
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

# --- 8. 修改主函数 ---
def main():
    """主函数"""
    try:
        print("=== Local LC25000 Lung Cancer Classification ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {DEVICE}")
        print(f"Local dataset path: {LOCAL_DATASET_PATH}")
        print()

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
        result = train_model()
        
        if result is not None:
            model, class_to_idx, idx_to_class,dataloaders = result
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
    main()