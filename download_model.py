import os
# 1. 设置环境变量，使用国内镜像站 (hf-mirror.com)
# 这行必须在 import huggingface_hub 之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print(">>> 正在使用 HF 镜像站下载 Qwen2.5-1.5B ...")

try:
    # 2. 下载模型
    # 这里的 repo_id 是 HuggingFace 上的 ID，和 ModelScope 基本一致
    model_dir = snapshot_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        local_dir="./local_models/Qwen2.5-1.5B-Instruct", # 指定下载到当前目录
        local_dir_use_symlinks=False, # 只要实体文件，不要软链接
        resume_download=True
    )
    print(f"\n✅ 模型下载成功！路径: {model_dir}")
    print("请在 robot_demo.py 中把 MODEL_PATH 改为上面的路径。")

except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("提示: 请先运行 pip install huggingface_hub")