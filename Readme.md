实验代码放在https://github.com/hxhxxxxx/Cancer_detect/blob/main/train_huggingface_cancer.py
需要把自己的数据集替换一下，文件格式详见：图片数据集文件夹示例.png


在运行代码里面需要手动修改的部分：
必须修改：
# 1. 修改数据集路径
LOCAL_DATASET_PATH = "/path/to/your/Gan"  # 改为你的肝癌数据集路径

# 2. 修改预定义类别（可选，主要用于文档说明）
CLASSES = ['ganyan', '正常']  # 肝癌、正常
NUM_CLASSES = len(CLASSES)

可选修改：
# 3. 修改项目描述
print("=== Local Liver Cancer Classification ===")  # 改为肝癌分类

# 4. 修改模型保存路径
BASE_DIR = "./liver_cancer_data"  # 改为肝癌相关
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_liver_model.pth")



最后就可以去用自己的数据训练该模型，结果类似于训练结果.png
对肺癌的检测准确率可以到达96%左右