import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- 配置参数 ---
# 模型路径
MODEL_PATH = "./lung_cancer_data/best_model.pth"

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

# --- 模型加载函数 ---
def load_inception_model():
    """加载Inception v3模型，兼容不同版本的torchvision"""
    try:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        print("Loaded model with new torchvision API")
    except AttributeError:
        model = models.inception_v3(pretrained=True)
        print("Loaded model with legacy torchvision API")
    return model

def load_trained_model(model_path):
    """
    加载训练好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # 创建模型结构（与训练时保持一致）
    model = load_inception_model()
    
    # 替换分类器
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 替换辅助分类器
    if hasattr(model, 'AuxLogits'):
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, NUM_CLASSES)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    
    # 移动到设备并设置为评估模式
    model = model.to(DEVICE)
    model.eval()
    
    print("Model loaded successfully!")
    return model

# --- 图像预处理函数 ---
def get_image_transform():
    """
    获取图像预处理变换（与训练时的验证变换保持一致）
    """
    return transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),  # Inception v3需要299x299
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path):
    """
    预处理输入图像
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
    
    # 应用预处理变换
    transform = get_image_transform()
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor

# --- 预测函数 ---
def predict_image(model, image_tensor):
    """
    使用模型对图像进行预测
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

def print_prediction_results(predicted_class, confidence, all_probabilities):
    """
    打印预测结果
    """
    print("\n" + "="*50)
    print("预测结果 / Prediction Results")
    print("="*50)
    
    # 主要预测结果
    class_name_cn = CLASS_NAMES_CN.get(predicted_class, predicted_class)
    print(f"预测类别: {class_name_cn} ({predicted_class})")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print("\n所有类别的概率分布:")
    print("-" * 30)
    for i, (class_name, prob) in enumerate(zip(CLASSES, all_probabilities)):
        class_name_cn = CLASS_NAMES_CN.get(class_name, class_name)
        print(f"{class_name_cn:8s} ({class_name:8s}): {prob:.4f} ({prob*100:.2f}%)")
    
    print("="*50)

# --- 主函数 ---
def main(image_path):
    """
    主预测函数
    """
    try:
        # 1. 加载训练好的模型
        print("Step 1: Loading trained model...")
        model = load_trained_model(MODEL_PATH)
        
        # 2. 预处理图像
        print("\nStep 2: Preprocessing image...")
        image_tensor = preprocess_image(image_path)
        
        # 3. 进行预测
        print("\nStep 3: Making prediction...")
        predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
        
        # 4. 显示结果
        print_prediction_results(predicted_class, confidence, all_probabilities)
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# --- 批量预测函数 ---
def predict_multiple_images(image_paths):
    """
    对多张图片进行批量预测
    """
    print("Loading model for batch prediction...")
    model = load_trained_model(MODEL_PATH)
    
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")
        try:
            image_tensor = preprocess_image(image_path)
            predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
            print_prediction_results(predicted_class, confidence, all_probabilities)
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

# --- 使用示例 ---
if __name__ == "__main__":
    # 单张图片预测示例
    # 请将下面的路径替换为您要预测的图片路径
    image_path = "/Users/huangxh/Documents/DMECL/LC25000/lung_aca/lungaca2.jpeg"  # 修改为实际图片路径
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please make sure you have trained the model and saved it to the correct location.")
    else:
        # 进行预测
        predicted_class, confidence = main(image_path)
        
        if predicted_class:
            print(f"\n最终结果: {CLASS_NAMES_CN.get(predicted_class, predicted_class)} (置信度: {confidence:.2f})")
    
    # 批量预测示例（可选）
    # image_list = ["/Users/huangxh/Documents/DMECL/LC25000/lung_aca/lungaca8.jpeg", "/Users/huangxh/Documents/DMECL/LC25000/lung_n/lungn10.jpeg", "/Users/huangxh/Documents/DMECL/LC25000/lung_scc/lungscc14.jpeg"]
    # results = predict_multiple_images(image_list)