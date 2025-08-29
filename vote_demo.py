import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import Counter
import warnings
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType
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

# 类别定义（与训练时保持一致）
CLASSES = ['lung_aca', 'lung_n', 'lung_scc']  # 肺腺癌、正常、肺鳞癌
NUM_CLASSES = len(CLASSES)

# 类别中文名称映射
CLASS_NAMES_CN = {
    'lung_aca': '肺腺癌',
    'lung_n': '正常',
    'lung_scc': '肺鳞癌'
}

# 类别详细描述
CLASS_DESCRIPTIONS = {
    'lung_aca': {
        'name_cn': '肺腺癌',
        'name_en': 'Lung Adenocarcinoma',
        'description': '肺腺癌是肺癌的一种类型，通常发生在肺的外周部位'
    },
    'lung_n': {
        'name_cn': '正常',
        'name_en': 'Normal',
        'description': '正常的肺部组织，无病理改变'
    },
    'lung_scc': {
        'name_cn': '肺鳞癌',
        'name_en': 'Lung Squamous Cell Carcinoma',
        'description': '肺鳞状细胞癌，通常发生在肺的中央部位'
    }
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

# --- 模型定义（从vote_fix.py复制） ---
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

def smart_load_checkpoint(model, checkpoint_path, model_name):
    """智能加载checkpoint，自动处理不同的保存格式"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 检查checkpoint的格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 尝试加载state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"⚠️  Missing keys in {model_name}: {len(missing_keys)} keys")
        if len(unexpected_keys) > 0:
            print(f"⚠️  Unexpected keys in {model_name}: {len(unexpected_keys)} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")
        return False

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

# --- 投票分类器类 ---
class LungCancerEnsembleClassifier:
    """肺癌多模型投票分类器"""
    
    def __init__(self, model_paths=MODEL_PATHS, verbose=True):
        self.device = DEVICE
        self.models = {}
        self.transforms = get_transforms()
        self.clip_processor = None
        self.verbose = verbose
        
        if self.verbose:
            print(f"🔧 Initializing Lung Cancer Ensemble Classifier")
            print(f"📱 Using device: {self.device}")
        
        # 加载所有模型
        self._load_models(model_paths)
    
    def _load_models(self, model_paths):
        """加载所有模型"""
        if self.verbose:
            print("\n📦 Loading models...")
        
        loaded_count = 0
        
        # 加载Inception模型
        if os.path.exists(model_paths['inception']):
            if self.verbose:
                print("  🔄 Loading Inception v3 model...")
            try:
                self.models['inception'] = create_inception_model()
                success = smart_load_checkpoint(
                    self.models['inception'], 
                    model_paths['inception'], 
                    'Inception'
                )
                if success:
                    self.models['inception'] = self.models['inception'].to(self.device)
                    self.models['inception'].eval()
                    loaded_count += 1
                    if self.verbose:
                        print("    ✅ Inception v3 loaded successfully")
                else:
                    del self.models['inception']
            except Exception as e:
                if self.verbose:
                    print(f"    ❌ Failed to load Inception model: {e}")
        else:
            if self.verbose:
                print(f"    ⚠️  Inception model not found: {model_paths['inception']}")
        
        # 加载ResNet50模型
        if os.path.exists(model_paths['resnet']):
            if self.verbose:
                print("  🔄 Loading ResNet50 model...")
            try:
                self.models['resnet'] = ResNet50Classifier()
                success = smart_load_checkpoint(
                    self.models['resnet'], 
                    model_paths['resnet'], 
                    'ResNet50'
                )
                if success:
                    self.models['resnet'] = self.models['resnet'].to(self.device)
                    self.models['resnet'].eval()
                    loaded_count += 1
                    if self.verbose:
                        print("    ✅ ResNet50 loaded successfully")
                else:
                    del self.models['resnet']
            except Exception as e:
                if self.verbose:
                    print(f"    ❌ Failed to load ResNet50 model: {e}")
        else:
            if self.verbose:
                print(f"    ⚠️  ResNet50 model not found: {model_paths['resnet']}")
        
        # 加载CLIP+LoRA模型
        if os.path.exists(model_paths['clip_lora']):
            if self.verbose:
                print("  🔄 Loading CLIP+LoRA model...")
            try:
                self.models['clip_lora'] = CLIPLoRAClassifier()
                success = smart_load_checkpoint(
                    self.models['clip_lora'], 
                    model_paths['clip_lora'], 
                    'CLIP+LoRA'
                )
                if success:
                    self.models['clip_lora'] = self.models['clip_lora'].to(self.device)
                    self.models['clip_lora'].eval()
                    self.clip_processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_MODEL_PATH)
                    loaded_count += 1
                    if self.verbose:
                        print("    ✅ CLIP+LoRA loaded successfully")
                else:
                    del self.models['clip_lora']
            except Exception as e:
                if self.verbose:
                    print(f"    ❌ Failed to load CLIP+LoRA model: {e}")
        else:
            if self.verbose:
                print(f"    ⚠️  CLIP+LoRA model not found: {model_paths['clip_lora']}")
        
        if self.verbose:
            print(f"\n📊 Successfully loaded {loaded_count}/3 models: {list(self.models.keys())}")
        
        if len(self.models) == 0:
            raise RuntimeError("❌ No models were successfully loaded! Please check your model paths and files.")
    
    def _preprocess_image(self, image_path, model_name):
        """为特定模型预处理图像"""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
        
        if model_name == 'clip_lora' and self.clip_processor is not None:
            # CLIP模型使用processor
            processed = self.clip_processor(images=image, return_tensors="pt")
            return processed['pixel_values'].to(self.device)
        else:
            # Inception和ResNet使用transforms
            transform = self.transforms[model_name]
            image_tensor = transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
    
    def _predict_single_model(self, image_tensor, model_name):
        """使用单个模型进行预测"""
        if model_name not in self.models:
            return None, None, None
        
        model = self.models[model_name]
        
        with torch.no_grad():
            # 特殊处理Inception模型的训练模式输出
            if model_name == 'inception' and hasattr(model, 'AuxLogits'):
                model.eval()  # 确保在评估模式
                outputs = model(image_tensor)
                # 在评估模式下，Inception只返回主输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取主输出
            else:
                outputs = model(image_tensor)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predicted_class_idx = predicted.item()
            predicted_class = CLASSES[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def predict(self, image_path, voting_method='majority'):
        """对单张图像进行投票预测"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if self.verbose:
            print(f"\n🔍 Analyzing image: {os.path.basename(image_path)}")
        
        predictions = {}
        all_probabilities = {}
        
        # 获取每个模型的预测
        for model_name in self.models.keys():
            if self.verbose:
                print(f"  🤖 Running {model_name} model...")
            
            # 预处理图像
            image_tensor = self._preprocess_image(image_path, model_name)
            
            # 预测
            pred_class, confidence, probs = self._predict_single_model(image_tensor, model_name)
            if pred_class is not None:
                predictions[model_name] = {
                    'class': pred_class,
                    'confidence': confidence,
                    'probabilities': probs
                }
                all_probabilities[model_name] = probs
                
                if self.verbose:
                    class_name_cn = CLASS_NAMES_CN.get(pred_class, pred_class)
                    print(f"    📊 {model_name}: {class_name_cn} ({confidence:.3f})")
        
        # 投票决策
        if not predictions:
            raise RuntimeError("❌ No valid predictions from any model")
        
        final_prediction = self._vote(predictions, voting_method)
        
        return final_prediction, predictions, all_probabilities
    
    def _vote(self, predictions, method='majority'):
        """投票决策"""
        if method == 'majority':
            # 简单多数投票
            votes = [pred['class'] for pred in predictions.values()]
            vote_counts = Counter(votes)
            final_class = vote_counts.most_common(1)[0][0]
            
            # 计算平均置信度
            class_confidences = [pred['confidence'] for pred in predictions.values() 
                               if pred['class'] == final_class]
            avg_confidence = np.mean(class_confidences)
            
            return {
                'final_class': final_class,
                'confidence': avg_confidence,
                'vote_counts': dict(vote_counts),
                'method': method
            }
        
        elif method == 'weighted':
            # 基于置信度的加权投票
            weighted_probs = np.zeros(NUM_CLASSES)
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = pred['confidence']
                weighted_probs += pred['probabilities'] * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_probs /= total_weight
            
            final_class_idx = np.argmax(weighted_probs)
            final_class = CLASSES[final_class_idx]
            final_confidence = weighted_probs[final_class_idx]
            
            return {
                'final_class': final_class,
                'confidence': final_confidence,
                'weighted_probabilities': weighted_probs,
                'method': method
            }

# --- 结果显示函数 ---
def print_prediction_results(final_prediction, individual_predictions, image_path):
    """打印详细的预测结果"""
    print("\n" + "="*70)
    print("🫁 肺癌智能诊断系统 - 预测结果 / Lung Cancer AI Diagnosis - Results")
    print("="*70)
    
    # 图像信息
    print(f"📁 图像文件: {os.path.basename(image_path)}")
    print(f"📍 完整路径: {image_path}")
    
    # 个体模型预测结果
    print(f"\n🤖 各模型预测结果:")
    print("-" * 50)
    for model_name, pred in individual_predictions.items():
        class_info = CLASS_DESCRIPTIONS[pred['class']]
        print(f"  {model_name:12s}: {class_info['name_cn']:6s} ({pred['class']:8s}) - 置信度: {pred['confidence']:.3f}")
    
    # 投票结果
    print(f"\n🗳️  集成投票结果:")
    print("-" * 50)
    final_class = final_prediction['final_class']
    final_confidence = final_prediction['confidence']
    class_info = CLASS_DESCRIPTIONS[final_class]
    
    print(f"🎯 最终预测: {class_info['name_cn']} ({class_info['name_en']})")
    print(f"📊 综合置信度: {final_confidence:.3f}")
    print(f"🔬 投票方法: {final_prediction['method']}")
    
    if 'vote_counts' in final_prediction:
        print(f"📈 投票统计: {final_prediction['vote_counts']}")
    
    # 类别描述
    print(f"\n📝 诊断说明:")
    print(f"   {class_info['description']}")
    
    # 置信度解释
    confidence_level = "高" if final_confidence > 0.8 else "中" if final_confidence > 0.6 else "低"
    print(f"\n⚡ 置信度评估: {confidence_level}置信度")
    
    if final_confidence > 0.8:
        print("   ✅ 模型对此预测非常确信")
    elif final_confidence > 0.6:
        print("   ⚠️  模型对此预测较为确信，建议结合其他检查")
    else:
        print("   ⚠️  模型对此预测不够确信，强烈建议进一步检查")
    
    print("="*70)
    print("⚠️  免责声明: 此结果仅供参考，不能替代专业医学诊断")
    print("="*70)

def print_welcome():
    """打印欢迎信息"""
    print("="*70)
    print("🫁 肺癌智能诊断系统 / Lung Cancer AI Diagnosis System")
    print("="*70)
    print("📋 支持的图像格式: .jpg, .jpeg, .png, .bmp, .tiff")
    print("🤖 集成模型: Inception v3 + ResNet50 + CLIP+LoRA")
    print("🗳️  投票方法: majority (多数投票) / weighted (加权投票)")
    print("="*70)

# --- 主要功能函数 ---
def predict_single_image(image_path, voting_method='majority', verbose=True):
    """预测单张图像"""
    try:
        # 初始化分类器
        classifier = LungCancerEnsembleClassifier(verbose=verbose)
        
        # 进行预测
        start_time = time.time()
        final_pred, individual_preds, all_probs = classifier.predict(image_path, voting_method)
        end_time = time.time()
        
        # 显示结果
        if verbose:
            print_prediction_results(final_pred, individual_preds, image_path)
            print(f"\n⏱️  预测耗时: {end_time - start_time:.2f} 秒")
        
        return final_pred, individual_preds
        
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {e}")
        return None, None

def interactive_mode():
    """交互模式"""
    print_welcome()
    
    classifier = None
    
    while True:
        print(f"\n{'='*50}")
        print("🎮 交互模式 - 请选择操作:")
        print("1. 预测图像 (多数投票)")
        print("2. 预测图像 (加权投票)")
        print("3. 退出")
        print("="*50)
        
        choice = input("请输入选项 (1-3): ").strip()
        
        if choice == '3':
            print("👋 感谢使用肺癌智能诊断系统!")
            break
        elif choice in ['1', '2']:
            image_path = input("请输入图像路径: ").strip().strip('"\'')
            
            if not os.path.exists(image_path):
                print(f"❌ 文件不存在: {image_path}")
                continue
            
            voting_method = 'majority' if choice == '1' else 'weighted'
            
            try:
                # 延迟初始化分类器
                if classifier is None:
                    print("\n🔧 首次使用，正在初始化模型...")
                    classifier = LungCancerEnsembleClassifier(verbose=True)
                
                # 进行预测
                start_time = time.time()
                final_pred, individual_preds, _ = classifier.predict(image_path, voting_method)
                end_time = time.time()
                
                # 显示结果
                print_prediction_results(final_pred, individual_preds, image_path)
                print(f"\n⏱️  预测耗时: {end_time - start_time:.2f} 秒")
                
            except Exception as e:
                print(f"❌ 预测过程中发生错误: {e}")
        else:
            print("❌ 无效选项，请重新选择")

# --- 命令行接口 ---
def main():
    parser = argparse.ArgumentParser(
        description="肺癌智能诊断系统 - Lung Cancer AI Diagnosis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python demo.py --image /path/to/image.jpg                    # 使用多数投票
  python demo.py --image /path/to/image.jpg --method weighted  # 使用加权投票
  python demo.py --interactive                                 # 交互模式
        """
    )
    
    parser.add_argument('--image', '-i', type=str, help='输入图像路径')
    parser.add_argument('--method', '-m', choices=['majority', 'weighted'], 
                       default='majority', help='投票方法 (默认: majority)')
    parser.add_argument('--interactive', action='store_true', help='启动交互模式')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式，减少输出')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ 图像文件不存在: {args.image}")
            sys.exit(1)
        
        if not args.quiet:
            print_welcome()
        
        predict_single_image(args.image, args.method, verbose=not args.quiet)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()