import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# --- 配置参数 ---
# 模型路径
MODEL_PATH = "./lung_cancer_data/clip_lora_best_model.pth"
LORA_WEIGHTS_PATH = "./lung_cancer_data/lora_weights"

# CLIP模型配置（与训练时保持一致）
# CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 建议使用在线模型避免连接问题
# 如果你有完整的本地模型，可以使用：
CLIP_MODEL_NAME = "/Users/huangxh/Documents/DMECL/clip-vit-base-patch32-local"

# 优化后的LoRA配置参数（与训练时保持一致）
LORA_CONFIG = {
    "r": 32,  # 增加LoRA rank
    "lora_alpha": 64,  # 增加alpha值
    "lora_dropout": 0.05,  # 降低dropout
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "out_proj",  # attention layers
        "fc1", "fc2"  # feed forward layers
    ]
}

# 类别定义（与训练时保持一致）
CLASSES = ['lung_aca', 'lung_n', 'lung_scc']  # 肺腺癌、正常、肺鳞癌
NUM_CLASSES = len(CLASSES)

# 类别中文名称映射
CLASS_NAMES_CN = {
    'lung_aca': '肺腺癌',
    'lung_n': '正常',
    'lung_scc': '肺鳞癌'
}

# 设备检测函数（与训练脚本保持一致）
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
print(f"Using device: {DEVICE}")

# --- 增强的CLIP + LoRA 模型定义（与训练脚本保持一致）---
class CLIPLoRAClassifier(nn.Module):
    """
    CLIP Vision Encoder + LoRA + 增强分类头模型
    """
    def __init__(self, clip_model_name=CLIP_MODEL_NAME, num_classes=NUM_CLASSES, 
                 lora_config=LORA_CONFIG, load_lora_from_path=None):
        super(CLIPLoRAClassifier, self).__init__()
        
        # 加载基础CLIP模型
        print("Loading base CLIP model...")
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
        
        # 应用LoRA到CLIP vision model
        if load_lora_from_path and os.path.exists(load_lora_from_path):
            print(f"Loading LoRA weights from: {load_lora_from_path}")
            self.clip_model = PeftModel.from_pretrained(self.base_clip_model, load_lora_from_path)
        else:
            print("Applying LoRA to CLIP vision model...")
            self.clip_model = get_peft_model(self.base_clip_model, self.lora_config)
        
        # 获取CLIP vision encoder的输出维度
        self.clip_hidden_size = self.base_clip_model.config.vision_config.hidden_size
        
        # 增强的分类头 - 与训练时保持一致
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
        
        print(f"Enhanced CLIP + LoRA hidden size: {self.clip_hidden_size}")
        print(f"Enhanced Classification head: {self.clip_hidden_size} -> 1024 -> 512 -> 256 -> {num_classes}")
    
    def forward(self, pixel_values):
        # 通过CLIP vision encoder (with LoRA) 提取特征
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        
        # 获取pooled输出 (CLS token的表示)
        pooled_output = vision_outputs.pooler_output  # [batch_size, hidden_size]
        
        # 通过增强的分类头
        logits = self.classifier(pooled_output)
        
        return logits

# --- 智能路径检测函数 ---
def get_clip_model_path():
    """
    获取CLIP模型路径，优先使用本地，回退到在线
    """
    local_path = "/Users/huangxh/Documents/DMECL/clip-vit-base-patch32-local"
    online_path = "openai/clip-vit-base-patch32"
    
    # 检查本地路径是否存在且包含必要文件
    if os.path.exists(local_path):
        config_file = os.path.join(local_path, "config.json")
        preprocessor_file = os.path.join(local_path, "preprocessor_config.json")
        
        if os.path.exists(config_file) and os.path.exists(preprocessor_file):
            print(f"✓ Using local CLIP model: {local_path}")
            return local_path
        else:
            print(f"⚠️ Local path exists but missing config files")
    
    print(f"✓ Using online CLIP model: {online_path}")
    print("Note: First run will download and cache the model")
    return online_path

# --- 模型加载函数 ---
def load_trained_clip_lora_model(model_path, lora_path=None):
    """
    加载训练好的增强CLIP + LoRA模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading Enhanced CLIP + LoRA model from: {model_path}")
    
    # 获取CLIP模型路径
    clip_model_name = get_clip_model_path()
    
    # 创建模型结构（与训练时保持一致）
    model = CLIPLoRAClassifier(
        clip_model_name=clip_model_name,
        num_classes=NUM_CLASSES,
        lora_config=LORA_CONFIG,
        load_lora_from_path=lora_path  # 如果提供LoRA路径，直接加载
    )
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    
    # 移动到设备并设置为评估模式
    model = model.to(DEVICE)
    model.eval()
    
    print("Enhanced CLIP + LoRA model loaded successfully!")
    return model

# --- 图像预处理函数 ---
def get_clip_processor():
    """
    获取CLIP处理器（与训练时保持一致）
    """
    clip_model_name = get_clip_model_path()
    return CLIPProcessor.from_pretrained(clip_model_name)

def get_clip_transforms():
    """
    获取CLIP的数据增强变换（用于推理时的额外预处理）
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # CLIP使用224x224
        # 注意：不在这里转换为tensor，因为CLIPProcessor会处理
    ])

def preprocess_image(image_path):
    """
    预处理输入图像（使用CLIP处理器）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path}")
        print(f"Original image size: {image.size}")
    except Exception as e:
        raise ValueError(f"Cannot load image {image_path}: {e}")
    
    # 应用基本变换（可选的数据增强）
    transform = get_clip_transforms()
    image = transform(image)
    
    # 使用CLIP处理器进行最终预处理
    processor = get_clip_processor()
    processed = processor(images=image, return_tensors="pt")
    image_tensor = processed['pixel_values']  # [1, 3, 224, 224]
    
    return image_tensor

# --- 测试时增强 (TTA) 预测函数 ---
def predict_with_tta(model, image_tensor, num_tta=8):
    """
    使用测试时增强提高预测稳定性和准确性
    """
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        # 原始预测
        outputs = model(image_tensor.to(DEVICE))
        all_outputs.append(outputs)
        
        # TTA预测
        for _ in range(num_tta):
            # 创建变换后的图像副本
            tta_image = image_tensor.clone()
            
            # 随机水平翻转
            if torch.rand(1) > 0.5:
                tta_image = torch.flip(tta_image, dims=[3])
            
            # 随机垂直翻转（医学图像适用）
            if torch.rand(1) > 0.7:
                tta_image = torch.flip(tta_image, dims=[2])
            
            # 轻微的亮度调整
            if torch.rand(1) > 0.6:
                brightness_factor = 0.9 + torch.rand(1) * 0.2  # 0.9-1.1
                tta_image = tta_image * brightness_factor
                tta_image = torch.clamp(tta_image, 0, 1)
            
            outputs = model(tta_image.to(DEVICE))
            all_outputs.append(outputs)
    
    # 平均所有预测
    avg_outputs = torch.stack(all_outputs).mean(dim=0)
    probabilities = torch.nn.functional.softmax(avg_outputs, dim=1)
    
    _, predicted = torch.max(avg_outputs, 1)
    predicted_class_idx = predicted.item()
    predicted_class = CLASSES[predicted_class_idx]
    confidence = probabilities[0][predicted_class_idx].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# --- 标准预测函数 ---
def predict_image(model, image_tensor):
    """
    使用增强CLIP + LoRA模型对图像进行预测
    """
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        # 前向传播
        outputs = model(image_tensor)
        
        # 获取预测概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 获取预测类别
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()
        predicted_class = CLASSES[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

def print_prediction_results(predicted_class, confidence, all_probabilities, use_tta=False):
    """
    打印预测结果
    """
    tta_text = " (with TTA)" if use_tta else ""
    print("\n" + "="*60)
    print(f"Enhanced CLIP + LoRA 模型预测结果{tta_text}")
    print(f"Enhanced CLIP + LoRA Model Prediction Results{tta_text}")
    print("="*60)
    
    # 主要预测结果
    class_name_cn = CLASS_NAMES_CN.get(predicted_class, predicted_class)
    print(f"预测类别: {class_name_cn} ({predicted_class})")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print("\n所有类别的概率分布:")
    print("-" * 30)
    for i, (class_name, prob) in enumerate(zip(CLASSES, all_probabilities)):
        class_name_cn = CLASS_NAMES_CN.get(class_name, class_name)
        print(f"{class_name_cn:8s} ({class_name:8s}): {prob:.4f} ({prob*100:.2f}%)")
    
    print("="*60)

# --- 主函数 ---
def main(image_path, use_tta=True):
    """
    主预测函数
    """
    try:
        # 1. 检查LoRA权重是否存在
        lora_path = None
        if os.path.exists(LORA_WEIGHTS_PATH):
            lora_path = LORA_WEIGHTS_PATH
            print(f"✓ Found LoRA weights at: {lora_path}")
        else:
            print(f"⚠️ LoRA weights not found at: {LORA_WEIGHTS_PATH}")
            print("Will load LoRA weights from the main model file")
        
        # 2. 加载训练好的增强CLIP + LoRA模型
        print("Step 1: Loading trained Enhanced CLIP + LoRA model...")
        model = load_trained_clip_lora_model(MODEL_PATH, lora_path)
        
        # 3. 预处理图像
        print("\nStep 2: Preprocessing image with CLIP processor...")
        image_tensor = preprocess_image(image_path)
        
        # 4. 进行预测
        print(f"\nStep 3: Making prediction with Enhanced CLIP + LoRA model...")
        if use_tta:
            print("Using Test Time Augmentation (TTA) for improved accuracy...")
            predicted_class, confidence, all_probabilities = predict_with_tta(model, image_tensor)
        else:
            print("Using standard prediction...")
            predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
        
        # 5. 显示结果
        print_prediction_results(predicted_class, confidence, all_probabilities, use_tta)
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during Enhanced CLIP + LoRA prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# --- 批量预测函数 ---
def predict_multiple_images(image_paths, use_tta=True):
    """
    对多张图片进行批量预测
    """
    print("Loading Enhanced CLIP + LoRA model for batch prediction...")
    
    # 检查LoRA权重
    lora_path = LORA_WEIGHTS_PATH if os.path.exists(LORA_WEIGHTS_PATH) else None
    model = load_trained_clip_lora_model(MODEL_PATH, lora_path)
    
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")
        try:
            image_tensor = preprocess_image(image_path)
            
            if use_tta:
                predicted_class, confidence, all_probabilities = predict_with_tta(model, image_tensor)
            else:
                predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
            
            print_prediction_results(predicted_class, confidence, all_probabilities, use_tta)
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'predicted_class': None,
                'confidence': None,
                'error': str(e)
            })
    
    return results

# --- 模型比较函数 ---
def compare_predictions(image_path):
    """
    比较标准预测和TTA预测的结果
    """
    print("=== Enhanced Model Prediction Comparison ===")
    
    # 标准预测
    print("\n--- Standard Prediction ---")
    standard_class, standard_conf = main(image_path, use_tta=False)
    
    # TTA预测
    print("\n--- TTA Prediction ---")
    tta_class, tta_conf = main(image_path, use_tta=True)
    
    # 比较结果
    print("\n" + "="*50)
    print("预测结果比较 / Prediction Comparison")
    print("="*50)
    if standard_class and tta_class:
        print(f"标准预测: {CLASS_NAMES_CN.get(standard_class, standard_class)} (置信度: {standard_conf:.4f})")
        print(f"TTA预测:  {CLASS_NAMES_CN.get(tta_class, tta_class)} (置信度: {tta_conf:.4f})")
        
        if standard_class == tta_class:
            print("✓ 两种方法预测结果一致")
            if tta_conf > standard_conf:
                print(f"✓ TTA提高了置信度 (+{tta_conf - standard_conf:.4f})")
        else:
            print("⚠️ 两种方法预测结果不一致，建议使用TTA结果")
    
    return standard_class, standard_conf, tta_class, tta_conf

# --- 模型信息函数 ---
def print_model_info():
    """
    打印增强模型信息
    """
    print("=== Enhanced CLIP + LoRA Model Information ===")
    print(f"Base CLIP model: {get_clip_model_path()}")
    print(f"Enhanced LoRA configuration:")
    print(f"  - Rank (r): {LORA_CONFIG['r']}")
    print(f"  - Alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"  - Dropout: {LORA_CONFIG['lora_dropout']}")
    print(f"  - Target modules: {LORA_CONFIG['target_modules']}")
    print(f"Model weights: {MODEL_PATH}")
    print(f"LoRA weights: {LORA_WEIGHTS_PATH}")
    print(f"Classes: {CLASSES}")
    print(f"Enhanced Classification Head: 768 -> 1024 -> 512 -> 256 -> {NUM_CLASSES}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 打印模型信息
    print_model_info()
    
    # 单张图片预测示例
    image_path = "/Users/huangxh/Documents/DMECL/LC25000/lung_aca/lungaca2.jpeg"  # 修改为实际图片路径
    
    # 检查必要的库
    try:
        import transformers
        import peft
        print(f"\n✓ Transformers version: {transformers.__version__}")
        print(f"✓ PEFT version: {peft.__version__}")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("Please install: pip install transformers peft")
        exit(1)
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Enhanced CLIP + LoRA model file not found at {MODEL_PATH}")
        print("Please make sure you have trained the Enhanced CLIP + LoRA model and saved it to the correct location.")
    else:
        # 进行预测比较
        print("\n" + "="*70)
        print("Enhanced CLIP + LoRA 预测演示")
        print("="*70)
        
        # 比较标准预测和TTA预测
        standard_class, standard_conf, tta_class, tta_conf = compare_predictions(image_path)
        
        if tta_class:
            print(f"\n🎉 推荐结果 (TTA): {CLASS_NAMES_CN.get(tta_class, tta_class)} (置信度: {tta_conf:.4f})")
    
    # 批量预测示例（可选）
    print("\n" + "="*70)
    print("批量预测示例 / Batch Prediction Example")
    print("="*70)
    image_list = [
        "/Users/huangxh/Documents/DMECL/LC25000/lung_aca/lungaca8.jpeg", 
        "/Users/huangxh/Documents/DMECL/LC25000/lung_n/lungn10.jpeg", 
        "/Users/huangxh/Documents/DMECL/LC25000/lung_scc/lungscc14.jpeg"
    ]
    
    if os.path.exists(MODEL_PATH):
        print("使用TTA进行批量预测...")
        results = predict_multiple_images(image_list, use_tta=True)
        
        # 打印批量预测汇总
        print("\n" + "="*60)
        print("批量预测汇总 / Batch Prediction Summary (with TTA)")
        print("="*60)
        for result in results:
            if result.get('predicted_class'):
                class_cn = CLASS_NAMES_CN.get(result['predicted_class'], result['predicted_class'])
                print(f"{os.path.basename(result['image_path']):20s} -> {class_cn} ({result['confidence']:.4f})")
            else:
                print(f"{os.path.basename(result['image_path']):20s} -> Error: {result.get('error', 'Unknown')}")