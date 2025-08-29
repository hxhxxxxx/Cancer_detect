import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
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
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "829_resnet50_best_model.pth")

# 本地数据集路径
LOCAL_DATASET_PATH = "/Users/huangxh/Documents/DMECL/LC25000"

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

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

# --- 2. ResNet50 分类器模型定义 ---
class ResNet50Classifier(nn.Module):
    """
    ResNet50 + 自定义分类头模型
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        
        # 加载预训练的ResNet50
        print("Loading ResNet50 model...")
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 获取ResNet50的特征维度
        self.feature_dim = self.backbone.fc.in_features
        
        # 移除原始的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加自定义分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # 打印模型信息
        self._print_model_info()
        
    def _print_model_info(self):
        """打印模型参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n=== ResNet50 Model Information ===")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Classification head: {self.feature_dim} -> 512 -> 256 -> {NUM_CLASSES}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    def forward(self, x):
        # 通过ResNet50骨干网络提取特征
        features = self.backbone(x)
        
        # 通过分类头
        logits = self.classifier(features)
        
        return logits

# --- 3. 图像预处理函数 ---
def get_resnet_transforms():
    """
    获取ResNet50的图像预处理变换
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),  # ResNet50使用224x224
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    return data_transforms

# --- 4. 数据集类 ---
class LocalImageDataset(Dataset):
    """
    本地图像数据集类，适配ResNet50输入格式
    """
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        self.original_image_paths = image_paths.copy()
        self.original_labels = labels.copy()
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        
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
        
        # 处理标签
        if isinstance(label, str) and self.class_to_idx:
            label = self.class_to_idx.get(label, 0)
        
        return image, label

# --- 5. 数据加载函数（与之前保持一致） ---
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
    """创建本地数据集的数据加载器"""
    print("--- Creating data loaders ---")
    
    image_paths, labels, class_to_idx, idx_to_class = load_local_dataset()
    
    if len(image_paths) == 0:
        print("No valid images found.")
        return None, None, None, None, None, None
    
    data_splits = create_data_splits_local(image_paths, labels, class_to_idx)
    
    data_transforms = get_resnet_transforms()
    
    pytorch_datasets = {}
    for split in ['train', 'validation', 'test']:
        pytorch_datasets[split] = LocalImageDataset(
            data_splits[split]['image_paths'],
            data_splits[split]['labels'],
            data_transforms.get(split, data_transforms['validation']),
            class_to_idx
        )
    
    dataloaders = {}
    for split in ['train', 'validation', 'test']:
        dataloaders[split] = DataLoader(
            pytorch_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    dataset_sizes = {split: len(pytorch_datasets[split]) for split in ['train', 'validation', 'test']}
    
    return pytorch_datasets, dataloaders, dataset_sizes, class_to_idx, idx_to_class, data_splits

# --- 6. 模型相关函数 ---
def load_resnet50_model():
    """加载ResNet50分类模型"""
    print(f"Loading ResNet50 model...")
    model = ResNet50Classifier(
        num_classes=NUM_CLASSES,
        pretrained=True
    )
    return model

# --- 7. 测试函数 ---
def test_model(model, test_dataloader, dataset_size, class_to_idx, idx_to_class):
    """
    测试训练好的ResNet50模型
    """
    print("\n--- Testing ResNet50 model on test dataset ---")
    
    model.eval()
    
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # 用于计算每个类别的准确率
    class_correct = {i: 0 for i in range(len(class_to_idx))}
    class_total = {i: 0 for i in range(len(class_to_idx))}
    
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Testing ResNet50")
        
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
    
    print(f"\n=== ResNet50 Test Results ===")
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
def train_model():
    """使用ResNet50模型训练本地LC25000数据集"""
    print("\n--- Training with ResNet50 ---")
    print(f"Using device: {DEVICE}")

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

    # 加载ResNet50模型
    model = load_resnet50_model()
    model = model.to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 使用不同的学习率策略
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
            
            # 添加详细的类别统计
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
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # 保存模型
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
# --- 9. 主函数 ---
def main():
    """主函数"""
    try:
        print("=== ResNet50 LC25000 Lung Cancer Classification ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {DEVICE}")
        print(f"Local dataset path: {LOCAL_DATASET_PATH}")
        print()

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
                
                print(f"\n🎉 Final ResNet50 Test Accuracy: {test_acc:.4f}")
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