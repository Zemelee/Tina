from huggingface_hub import snapshot_download
import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

if __name__ == "__main__":
    CKPT_DIR = os.environ['CKPT_DIR']

    print("Downloading deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ...")
    snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      token="",
                      local_dir=f"{CKPT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/base")
