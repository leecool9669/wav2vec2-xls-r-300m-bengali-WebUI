import os
import pathlib
import urllib.request

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
HF_REPO_DIR = BASE_DIR / "hf_repo"
IMG_DIR = BASE_DIR / "images"

# 仅下载小文件；明确跳过大权重（如 pytorch_model.bin）
FILES = {
    ".gitattributes": "https://hf-mirror.com/arijitx/wav2vec2-xls-r-300m-bengali/resolve/main/.gitattributes",
    "config.json": "https://hf-mirror.com/arijitx/wav2vec2-xls-r-300m-bengali/resolve/main/config.json",
    "preprocessor_config.json": "https://hf-mirror.com/arijitx/wav2vec2-xls-r-300m-bengali/resolve/main/preprocessor_config.json",
    "tokenizer_config.json": "https://hf-mirror.com/arijitx/wav2vec2-xls-r-300m-bengali/resolve/main/tokenizer_config.json",
    "vocab.json": "https://hf-mirror.com/arijitx/wav2vec2-xls-r-300m-bengali/resolve/main/vocab.json",
}

IMAGES = {
    # 页面元信息中的社交缩略图（用于 README 与可视化展示）
    "model_social_thumbnail.png": "https://cdn-thumbnails.hf-mirror.com/social-thumbnails/models/arijitx/wav2vec2-xls-r-300m-bengali.png",
    # 作者头像（页面可见图片之一，jpeg）
    "author_avatar.jpeg": "https://cdn-avatars.hf-mirror.com/v1/production/uploads/1624653998840-60d48f9e50c47659f83f5ccf.jpeg",
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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
    except Exception as e:
        # 某些文件在仓库中可能不存在（例如 vocab.json/tokenizer_config.json），这里允许“尽力而为”。
        print(f"[skip] {url} -> {dest} ({e})")


def main() -> None:
    for name, url in FILES.items():
        download(url, HF_REPO_DIR / name)

    for name, url in IMAGES.items():
        download(url, IMG_DIR / name)


if __name__ == "__main__":
    main()
