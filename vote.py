import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import Counter
import pandas as pd
from tqdm import tqdm
import warnings
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType
import glob

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

# --- 1. Inception模型定义 - 修复版本 ---
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

# --- 2. ResNet50模型定义 ---
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

# --- 3. CLIP+LoRA模型定义 ---
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

# --- 4. 智能模型加载函数 ---
def smart_load_checkpoint(model, checkpoint_path, model_name):
    """智能加载checkpoint，自动处理不同的保存格式"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 检查checkpoint的格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # 格式: {"state_dict": model.state_dict()}
                print(f"Loading {model_name} with 'state_dict' format...")
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                # 格式: {"model_state_dict": model.state_dict()}
                print(f"Loading {model_name} with 'model_state_dict' format...")
                state_dict = checkpoint['model_state_dict']
            else:
                # 直接是state_dict格式
                print(f"Loading {model_name} with direct state_dict format...")
                state_dict = checkpoint
        else:
            # 可能是直接保存的模型
            print(f"Loading {model_name} with direct model format...")
            state_dict = checkpoint
        
        # 尝试加载state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"⚠️  Missing keys in {model_name}: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"⚠️  Unexpected keys in {model_name}: {unexpected_keys}")
        
        print(f"✅ {model_name} model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")
        return False

# --- 5. 图像预处理函数 ---
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

# --- 6. 多模型投票分类器 ---
class EnsembleVotingClassifier:
    """多模型投票分类器"""
    
    def __init__(self, model_paths=MODEL_PATHS):
        self.device = DEVICE
        self.models = {}
        self.transforms = get_transforms()
        
        # 只在需要时初始化CLIP processor
        self.clip_processor = None
        
        # 加载所有模型
        self._load_models(model_paths)
    
    def _load_models(self, model_paths):
        """加载所有模型"""
        print("Loading all models...")
        
        # 加载Inception模型 - 使用原生模型
        if os.path.exists(model_paths['inception']):
            print("Loading Inception model...")
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
                else:
                    del self.models['inception']
            except Exception as e:
                print(f"❌ Failed to load Inception model: {e}")
        else:
            print(f"❌ Inception model not found: {model_paths['inception']}")
        
        # 加载ResNet50模型
        if os.path.exists(model_paths['resnet']):
            print("Loading ResNet50 model...")
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
                else:
                    del self.models['resnet']
            except Exception as e:
                print(f"❌ Failed to load ResNet50 model: {e}")
        else:
            print(f"❌ ResNet50 model not found: {model_paths['resnet']}")
        
        # 加载CLIP+LoRA模型
        if os.path.exists(model_paths['clip_lora']):
            print("Loading CLIP+LoRA model...")
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
                    # 初始化CLIP processor
                    self.clip_processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_MODEL_PATH)
                else:
                    del self.models['clip_lora']
            except Exception as e:
                print(f"❌ Failed to load CLIP+LoRA model: {e}")
        else:
            print(f"❌ CLIP+LoRA model not found: {model_paths['clip_lora']}")
        
        print(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
        
        if len(self.models) == 0:
            raise RuntimeError("No models were successfully loaded! Please check your model paths and files.")
    
    def _preprocess_image(self, image_path, model_name):
        """为特定模型预处理图像"""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
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
    
    def predict_single_image(self, image_path, voting_method='majority', verbose=True):
        """对单张图像进行投票预测"""
        if verbose:
            print(f"\n=== Predicting: {os.path.basename(image_path)} ===")
        
        predictions = {}
        all_probabilities = {}
        
        # 获取每个模型的预测
        for model_name in self.models.keys():
            if verbose:
                print(f"Running {model_name} model...")
            
            # 预处理图像
            image_tensor = self._preprocess_image(image_path, model_name)
            if image_tensor is None:
                continue
            
            # 预测
            pred_class, confidence, probs = self._predict_single_model(image_tensor, model_name)
            if pred_class is not None:
                predictions[model_name] = {
                    'class': pred_class,
                    'confidence': confidence,
                    'probabilities': probs
                }
                all_probabilities[model_name] = probs
                
                if verbose:
                    class_name_cn = CLASS_NAMES_CN.get(pred_class, pred_class)
                    print(f"  {model_name}: {class_name_cn} ({pred_class}) - {confidence:.4f}")
        
        # 投票决策
        if not predictions:
            if verbose:
                print("❌ No valid predictions from any model")
            return None, None, None
        
        final_prediction = self._vote(predictions, voting_method, verbose)
        
        return final_prediction, predictions, all_probabilities
    
    def _vote(self, predictions, method='majority', verbose=True):
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
            
            if verbose:
                print(f"\n🗳️  Voting Results:")
                print(f"Vote counts: {dict(vote_counts)}")
                print(f"Final prediction: {CLASS_NAMES_CN.get(final_class, final_class)} ({final_class})")
                print(f"Average confidence: {avg_confidence:.4f}")
            
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
            
            if verbose:
                print(f"\n🗳️  Weighted Voting Results:")
                print(f"Final prediction: {CLASS_NAMES_CN.get(final_class, final_class)} ({final_class})")
                print(f"Weighted confidence: {final_confidence:.4f}")
            
            return {
                'final_class': final_class,
                'confidence': final_confidence,
                'weighted_probabilities': weighted_probs,
                'method': method
            }
    
    def predict_batch(self, image_paths, voting_method='majority', save_results=False):
        """批量预测 - 修改为不保存CSV，只显示准确率统计"""
        print(f"\n=== Batch Prediction: {len(image_paths)} images ===")
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        # 按类别统计
        class_stats = {class_name: {'correct': 0, 'total': 0} for class_name in CLASSES}
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                final_pred, individual_preds, all_probs = self.predict_single_image(
                    image_path, voting_method, verbose=False  # 批量处理时不显示详细信息
                )
                
                if final_pred is not None:
                    # 从文件路径推断真实标签（如果可能）
                    true_label = self._extract_true_label(image_path)
                    
                    if true_label:
                        class_stats[true_label]['total'] += 1
                        total_predictions += 1
                        
                        if final_pred['final_class'] == true_label:
                            class_stats[true_label]['correct'] += 1
                            correct_predictions += 1
                    
                    # 简化的结果记录（不保存详细信息）
                    result = {
                        'true_label': true_label,
                        'predicted_label': final_pred['final_class'],
                        'correct': final_pred['final_class'] == true_label if true_label else None
                    }
                    results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # 打印详细的准确率统计
        self._print_accuracy_stats(class_stats, total_predictions, correct_predictions, voting_method)
        
        return results
    
    def _print_accuracy_stats(self, class_stats, total_predictions, correct_predictions, voting_method):
        """打印详细的准确率统计"""
        print(f"\n" + "="*60)
        print(f"📊 {voting_method.upper()} VOTING - ACCURACY RESULTS")
        print(f"="*60)
        
        # 总体准确率
        if total_predictions > 0:
            overall_accuracy = correct_predictions / total_predictions
            print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        else:
            print("🎯 Overall Accuracy: No valid predictions")
        
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
    
    def _extract_true_label(self, image_path):
        """从文件路径提取真实标签"""
        for class_name in CLASSES:
            if class_name in image_path:
                return class_name
        return None

# --- 7. 修改后的主函数 ---
def main():
    """主函数 - 修改为处理所有LC25000数据"""
    try:
        # 创建投票分类器
        ensemble = EnsembleVotingClassifier()
        
        # 收集所有测试图片
        test_images = []
        dataset_path = "/Users/huangxh/Documents/DMECL/LC25000"
        
        if os.path.exists(dataset_path):
            print(f"📁 Loading all images from {dataset_path}")
            
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
            
            print(f"\n🔍 Total images to process: {len(test_images)}")
            
            if test_images:
                # 简单多数投票
                print(f"\n{'='*60}")
                print(f"🗳️  Running MAJORITY VOTING on all {len(test_images)} images")
                print(f"{'='*60}")
                
                results_majority = ensemble.predict_batch(
                    test_images, voting_method='majority', save_results=False
                )
                
                # 加权投票
                print(f"\n{'='*60}")
                print(f"🗳️  Running WEIGHTED VOTING on all {len(test_images)} images")
                print(f"{'='*60}")
                
                results_weighted = ensemble.predict_batch(
                    test_images, voting_method='weighted', save_results=False
                )
                
                # 比较两种投票方法
                print(f"\n{'='*60}")
                print(f"📊 VOTING METHODS COMPARISON")
                print(f"{'='*60}")
                print(f"Total images processed: {len(test_images)}")
                print(f"Majority voting and weighted voting results shown above.")
                
            else:
                print("❌ No images found in the dataset!")
        else:
            print(f"❌ Dataset path not found: {dataset_path}")
            
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()