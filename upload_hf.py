import os
from huggingface_hub import HfApi, HfFolder, upload_folder

# ✅ 设置你的 HF Token（最好用 write 权限的 token）
HF_TOKEN = ""
HfFolder.save_token(HF_TOKEN)
api = HfApi()

# ✅ 本地模型文件夹列表（可以根据实际路径添加）
model_paths = [

    "./opencompass/outputs/Rebuttal-0727"
]

# ✅ 你的 HF 用户名或组织名
HF_USERNAME = "yang31210999"  # 如果是组织就写 org 名

for path in model_paths:
    # model_name = os.path.basename(path)  # e.g. Qwen3-0.6B-AWQ-2b
    model_name="Rebuttal-0727_OC-H200-d1"
    repo_id = f"{HF_USERNAME}/{model_name}"
    print(f"🚀 正在上传 {model_name} 到 {repo_id} ...")

    # ✅ 如果仓库不存在，就创建
    try:
        api.create_repo(repo_id, private=False)  # 改成 private=True 可以建私有仓
    except Exception as e:
        print(f"⚠️ 仓库 {repo_id} 已存在，跳过创建")

    # ✅ 上传整个文件夹
    upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model weights"
    )

print("✅ 全部上传完成！")
