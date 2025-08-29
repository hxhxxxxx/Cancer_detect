from huggingface_hub import snapshot_download
import os

# --- 配置 ---
# 1. 你想要下载的模型的ID
repo_id = "openai/clip-vit-base-patch32"

# 2. 你想把模型文件存放在本地的哪个文件夹
#    建议使用一个能清楚表明模型身份的文件夹名
local_dir = "./clip-vit-base-patch32-local"

# 3. (可选，但对您极其重要！) 设置镜像，解决网络问题
#    确保您已经按照之前的建议，在终端设置了环境变量
#    export HF_ENDPOINT=https://hf-mirror.com
#    脚本会自动使用这个镜像地址

print(f"正在从 Hugging Face Hub 下载模型: {repo_id}")
print(f"将保存到本地目录: {local_dir}")

# --- 执行下载 ---
# snapshot_download 会下载仓库中的所有文件
# local_dir_use_symlinks="auto" 是一个很好的实践，它会优先尝试使用符号链接以节省磁盘空间，
# 如果不支持或有权限问题，则会直接复制文件。
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks="auto" 
)

print("\n模型所有文件下载完毕！")
print(f"文件位于: {os.path.abspath(local_dir)}")
