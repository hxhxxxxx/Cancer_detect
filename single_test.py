import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType
import glob
import time

warnings.filterwarnings('ignore')

# --- 配置参数 ---
# 模型路径配置
MODEL_PATHS = {
    'inception': "./lung_cancer_data/829_best_model.pth",
    'resnet': "./lung_cancer_data/resnet50_best_model.pth", 
    'clip_lora': "./lung_cancer_data/clip_lora_best_model.pth"
}

# CLIP模型配置
LOCAL_CLIP_MODEL_PATH = "/Users/huangxh/Documents/DMECL/clip-vit-base-patch32-local"

# 类别定义（与训练时保
CLASSES = ['lung_aca', 'lung_n', 'lung_scc']  # 肺腺癌、正常、肺鳞癌
NUM_CLASSES = len(CLASSES)

# 类别中文名称映射
CLASS_NAMES_CN = {
    'lung_aca': '肺腺癌',
    'lung_n': '正常',
    'lung_scc': '肺鳞癌'
}

# LoRA配置（与训练时保持一致）
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2"
    ]
}

# 设备检测
def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# --- 模型定义（与vote_fix.py保持一致） ---
def load_inception_model():
    """加载Inception v3模型，兼容不同版本的torchvision"""
    try:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    except AttributeError:
        model = models.inception_v3(pretrained=True)
    return model

def create_inception_model(num_classes=NUM_CLASSES):
    """创建与训练时完全一致的Inception模型"""
    model = load_inception_model()
    
    # 替换分类器（与训练时保持一致）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 替换辅助分类器（与训练时保持一致）
    if hasattr(model, 'AuxLogits'):
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
    
    return model

class ResNet50Classifier(nn.Module):
    """ResNet50 + 自定义分类头模型"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super(ResNet50Classifier, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
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
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class CLIPLoRAClassifier(nn.Module):
    """CLIP Vision Encoder + LoRA + 增强分类头模型"""
    def __init__(self, clip_model_name=LOCAL_CLIP_MODEL_PATH, num_classes=NUM_CLASSES, 
                 lora_config=LORA_CONFIG):
        super(CLIPLoRAClassifier, self).__init__()
        
        self.base_clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 冻结所有CLIP参数
        for param in self.base_clip_model.parameters():
            param.requires_grad = False
        
        # 配置LoRA
        self.lora_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        self.clip_model = get_peft_model(self.base_clip_model, self.lora_config)
        self.clip_hidden_size = self.base_clip_model.config.vision_config.hidden_size
        
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
    
    def forward(self, pixel_values):
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# --- 智能模型加载函数 ---
def smart_load_checkpoint(model, checkpoint_path, model_name):
    """智能加载checkpoint，自动处理不同的保存格式"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 检查checkpoint的格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print(f"Loading {model_name} with 'state_dict' format...")
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                print(f"Loading {model_name} with 'model_state_dict' format...")
                state_dict = checkpoint['model_state_dict']
            else:
                print(f"Loading {model_name} with direct state_dict format...")
                state_dict = checkpoint
        else:
            print(f"Loading {model_name} with direct model format...")
            state_dict = checkpoint
        
        # 尝试加载state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"⚠️  Missing keys in {model_name}: {len(missing_keys)} keys")
        if len(unexpected_keys) > 0:
            print(f"⚠️  Unexpected keys in {model_name}: {len(unexpected_keys)} keys")
        
        print(f"✅ {model_name} model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")
        return False

# --- 图像预处理函数 ---
def get_transforms():
    """获取各模型的预处理变换"""
    transforms_dict = {
        'inception': transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'resnet': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'clip_lora': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    }
    return transforms_dict

# --- 单模型测试器类 ---
class SingleModelTester:
    """单模型测试器"""
    
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.device = DEVICE
        self.transforms = get_transforms()
        self.model = None
        self.clip_processor = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载指定的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"\n🔄 Loading {self.model_name} model...")
        
        try:
            if self.model_name == 'inception':
                self.model = create_inception_model()
            elif self.model_name == 'resnet':
                self.model = ResNet50Classifier()
            elif self.model_name == 'clip_lora':
                self.model = CLIPLoRAClassifier()
                self.clip_processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_MODEL_PATH)
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")
            
            # 加载权重
            success = smart_load_checkpoint(self.model, self.model_path, self.model_name)
            if not success:
                raise RuntimeError(f"Failed to load {self.model_name} model weights")
            
            # 移动到设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ {self.model_name} model loaded and ready for testing")
            
        except Exception as e:
            print(f"❌ Error loading {self.model_name} model: {e}")
            raise
    
    def _preprocess_image(self, image_path):
        """预处理图像"""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
        if self.model_name == 'clip_lora' and self.clip_processor is not None:
            # CLIP模型使用processor
            processed = self.clip_processor(images=image, return_tensors="pt")
            return processed['pixel_values'].to(self.device)
        else:
            # Inception和ResNet使用transforms
            transform = self.transforms[self.model_name]
            image_tensor = transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
    
    def _predict_single_image(self, image_tensor):
        """对单张图像进行预测"""
        with torch.no_grad():
            # 特殊处理Inception模型的训练模式输出
            if self.model_name == 'inception' and hasattr(self.model, 'AuxLogits'):
                self.model.eval()  # 确保在评估模式
                outputs = self.model(image_tensor)
                # 在评估模式下，Inception只返回主输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取主输出
            else:
                outputs = self.model(image_tensor)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predicted_class_idx = predicted.item()
            predicted_class = CLASSES[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def test_on_dataset(self, image_paths):
        """在数据集上测试模型"""
        print(f"\n{'='*70}")
        print(f"🧪 Testing {self.model_name.upper()} Model on {len(image_paths)} images")
        print(f"{'='*70}")
        
        correct_predictions = 0
        total_predictions = 0
        
        # 按类别统计
        class_stats = {class_name: {'correct': 0, 'total': 0} for class_name in CLASSES}
        
        # 记录预测时间
        start_time = time.time()
        
        for image_path in tqdm(image_paths, desc=f"Testing {self.model_name}"):
            try:
                # 预处理图像
                image_tensor = self._preprocess_image(image_path)
                if image_tensor is None:
                    continue
                
                # 预测
                pred_class, confidence, probs = self._predict_single_image(image_tensor)
                if pred_class is not None:
                    # 从文件路径推断真实标签
                    true_label = self._extract_true_label(image_path)
                    
                    if true_label:
                        class_stats[true_label]['total'] += 1
                        total_predictions += 1
                        
                        if pred_class == true_label:
                            class_stats[true_label]['correct'] += 1
                            correct_predictions += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        end_time = time.time()
        test_time = end_time - start_time
        
        # 打印详细的准确率统计
        self._print_test_results(class_stats, total_predictions, correct_predictions, test_time)
        
        return class_stats, total_predictions, correct_predictions
    
    def _extract_true_label(self, image_path):
        """从文件路径提取真实标签"""
        for class_name in CLASSES:
            if class_name in image_path:
                return class_name
        return None
    
    def _print_test_results(self, class_stats, total_predictions, correct_predictions, test_time):
        """打印测试结果"""
        print(f"\n📊 {self.model_name.upper()} MODEL - TEST RESULTS")
        print(f"="*60)
        
        # 总体准确率
        if total_predictions > 0:
            overall_accuracy = correct_predictions / total_predictions
            print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        else:
            print("🎯 Overall Accuracy: No valid predictions")
        
        print(f"⏱️  Test Time: {test_time:.2f} seconds")
        print(f"🚀 Speed: {total_predictions/test_time:.2f} images/second")
        
        print(f"\n📈 Accuracy by Class:")
        print("-" * 50)
        
        for class_name in CLASSES:
            stats = class_stats[class_name]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                class_name_cn = CLASS_NAMES_CN.get(class_name, class_name)
                print(f"  {class_name_cn:8s} ({class_name:8s}): {accuracy:.4f} ({stats['correct']:3d}/{stats['total']:3d})")
            else:
                class_name_cn = CLASS_NAMES_CN.get(class_name, class_name)
                print(f"  {class_name_cn:8s} ({class_name:8s}): No samples")
        
        print("="*60)

# --- 数据集加载函数 ---
def load_test_dataset():
    """加载测试数据集"""
    dataset_path = "/Users/huangxh/Documents/DMECL/LC25000"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    test_images = []
    
    print(f"📁 Loading test images from {dataset_path}")
    
    for class_name in CLASSES:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            # 获取该类别的所有图片
            class_images = []
            for ext in ['*.jpeg', '*.jpg', '*.png']:
                class_images.extend(glob.glob(os.path.join(class_dir, ext)))
            
            test_images.extend(class_images)
            class_name_cn = CLASS_NAMES_CN.get(class_name, class_name)
            print(f"  {class_name_cn} ({class_name}): {len(class_images)} images")
    
    print(f"\n🔍 Total images to test: {len(test_images)}")
    return test_images

# --- 比较结果函数 ---
def compare_model_results(results):
    """比较所有模型的测试结果"""
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON - SUMMARY RESULTS")
    print(f"{'='*80}")
    
    # 表头
    print(f"{'Model':<15} {'Overall Acc':<12} {'肺腺癌 Acc':<12} {'正常 Acc':<10} {'肺鳞癌 Acc':<12}")
    print("-" * 80)
    
    # 每个模型的结果
    for model_name, (class_stats, total_preds, correct_preds) in results.items():
        overall_acc = correct_preds / total_preds if total_preds > 0 else 0
        
        # 计算各类别准确率
        class_accs = {}
        for class_name in CLASSES:
            stats = class_stats[class_name]
            if stats['total'] > 0:
                class_accs[class_name] = stats['correct'] / stats['total']
            else:
                class_accs[class_name] = 0
        
        print(f"{model_name:<15} {overall_acc:<12.4f} {class_accs['lung_aca']:<12.4f} "
              f"{class_accs['lung_n']:<10.4f} {class_accs['lung_scc']:<12.4f}")
    
    print("="*80)
    
    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1][1]/x[1][2] if x[1][2] > 0 else 0)
    best_acc = best_model[1][1] / best_model[1][2] if best_model[1][2] > 0 else 0
    
    print(f"\n🏆 Best Overall Performance: {best_model[0].upper()} ({best_acc:.4f})")
    
    # 各类别最佳性能
    print(f"\n🎯 Best Performance by Class:")
    for class_name in CLASSES:
        class_name_cn = CLASS_NAMES_CN.get(class_name, class_name)
        best_class_model = None
        best_class_acc = 0
        
        for model_name, (class_stats, _, _) in results.items():
            stats = class_stats[class_name]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                if acc > best_class_acc:
                    best_class_acc = acc
                    best_class_model = model_name
        
        if best_class_model:
            print(f"  {class_name_cn}: {best_class_model.upper()} ({best_class_acc:.4f})")

# --- 主函数 ---
def main():
    """主函数"""
    try:
        print("="*80)
        print("🧪 Individual Model Testing on LC25000 Dataset")
        print("="*80)
        print(f"📱 Device: {DEVICE}")
        print(f"🎯 Models to test: {list(MODEL_PATHS.keys())}")
        
        # 加载测试数据集
        test_images = load_test_dataset()
        
        if len(test_images) == 0:
            print("❌ No test images found!")
            return
        
        # 存储所有模型的测试结果
        all_results = {}
        
        # 逐个测试每个模型
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                try:
                    # 创建单模型测试器
                    tester = SingleModelTester(model_name, model_path)
                    
                    # 在数据集上测试
                    class_stats, total_preds, correct_preds = tester.test_on_dataset(test_images)
                    
                    # 保存结果
                    all_results[model_name] = (class_stats, total_preds, correct_preds)
                    
                except Exception as e:
                    print(f"❌ Error testing {model_name}: {e}")
                    continue
            else:
                print(f"⚠️  Model file not found: {model_path}")
        
        # 比较所有模型的结果
        if all_results:
            compare_model_results(all_results)
        else:
            print("❌ No models were successfully tested!")
        
        print(f"\n✅ Individual model testing completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()