import os
import pathlib
import urllib.request

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
HF_REPO_DIR = BASE_DIR / "hf_repo"
IMG_DIR = BASE_DIR / "images"

# 仅下载小文件；明确跳过大权重（如 pytorch_model.bin）
FILES = {
    ".gitattributes": "https://hf-mirror.com/dmis-lab/biobert-base-cased-v1.1/resolve/main/.gitattributes",
    "config.json": "https://hf-mirror.com/dmis-lab/biobert-base-cased-v1.1/resolve/main/config.json",
    "vocab.txt": "https://hf-mirror.com/dmis-lab/biobert-base-cased-v1.1/resolve/main/vocab.txt",
}

IMAGES = {
    # 页面元信息中的社交缩略图（用于 README 与可视化展示）
    "model_social_thumbnail.png": "https://cdn-thumbnails.hf-mirror.com/social-thumbnails/models/dmis-lab/biobert-base-cased-v1.1.png",
    # 作者头像（页面可见图片之一）
    "author_avatar.png": "https://cdn-avatars.hf-mirror.com/v1/production/uploads/1602668910270-5efbdc4ac3896117eab961a9.png",
    # HF Mirror Logo（页面页眉图标；用于 README 标识来源）
    "hf_mirror_logo.svg": "https://hf-mirror.com/front/assets/huggingface_logo-noborder.svg",
}


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 使用系统代理（如用户配置 http://127.0.0.1:18081）
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    opener = urllib.request.build_opener()
    if proxy:
        opener.add_handler(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    urllib.request.install_opener(opener)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)


def main() -> None:
    for name, url in FILES.items():
        download(url, HF_REPO_DIR / name)

    for name, url in IMAGES.items():
        download(url, IMG_DIR / name)


if __name__ == "__main__":
    main()
