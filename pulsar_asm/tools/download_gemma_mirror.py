# ==============================================================================
# Project PULSAR-ASM | Tool: download_gemma_mirror.py
# ------------------------------------------------------------------------------
# Zero-Token Automated Downloader for Google Gemma-2B-IT from Open Mirror
# Features:
#   1. Zero Incantation (No HuggingFace token required)
#   2. HTTP Range resume support
#   3. Real-time bandwidth, ETA, and progress display
#   4. Downloads:
#      - model-00001-of-00002.safetensors (4.94 GB)
#      - model-00002-of-00002.safetensors (67.1 MB)
#      - tokenizer.model (4.24 MB)
# ==============================================================================

import os
import sys
import time
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "https://modelscope.cn/api/v1/models/AI-ModelScope/gemma-2b-it/repo?Revision=master&FilePath="

FILES_TO_DOWNLOAD = [
    ("tokenizer.model", 4241003),
    ("model-00002-of-00002.safetensors", 67121608),
    ("model-00001-of-00002.safetensors", 4945242264),
]

def format_bytes(n):
    if n >= 1024 ** 3:
        return f"{n / (1024 ** 3):.2f} GB"
    elif n >= 1024 ** 2:
        return f"{n / (1024 ** 2):.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"

def download_file_with_resume(filename, expected_size, target_dir):
    url = BASE_URL + filename
    target_path = os.path.join(target_dir, filename)
    os.makedirs(target_dir, exist_ok=True)

    existing_bytes = 0
    if os.path.exists(target_path):
        existing_bytes = os.path.getsize(target_path)
        if existing_bytes == expected_size:
            print(f"✔ [{filename}] 已存在且大小相符 ({format_bytes(existing_bytes)})，略過下載。")
            return target_path
        elif existing_bytes > expected_size:
            print(f"⚠ [{filename}] 檔案大小異常 ({existing_bytes} > {expected_size})，重新下載...")
            os.remove(target_path)
            existing_bytes = 0
        else:
            print(f"🔄 [{filename}] 偵測到斷點 ({format_bytes(existing_bytes)} / {format_bytes(expected_size)})，接續下載...")

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    if existing_bytes > 0:
        headers['Range'] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    block_size = 256 * 1024  # 256 KB chunk for socket buffer stability

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            mode = "ab" if existing_bytes > 0 else "wb"
            downloaded = existing_bytes
            t_start = time.perf_counter()
            t_last = t_start
            bytes_last = downloaded

            print(f"📥 開始下載: {filename} (總大小: {format_bytes(expected_size)})")

            with open(target_path, mode) as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    t_now = time.perf_counter()
                    if t_now - t_last >= 1.0 or downloaded == expected_size:
                        speed = (downloaded - bytes_last) / (t_now - t_last)
                        pct = (downloaded / expected_size) * 100
                        rem_bytes = expected_size - downloaded
                        eta_sec = rem_bytes / speed if speed > 0 else 0

                        speed_str = f"{format_bytes(speed)}/s"
                        progress_bar = ("#" * int(pct // 5)).ljust(20, ".")
                        print(
                            f"\r   [{progress_bar}] {pct:5.1f}% | "
                            f"{format_bytes(downloaded)} / {format_bytes(expected_size)} | "
                            f"{speed_str} | ETA: {eta_sec:.0f}s    ",
                            end="",
                            flush=True
                        )
                        t_last = t_now
                        bytes_last = downloaded

            print(f"\n✔ [{filename}] 下載完成！耗時: {time.perf_counter() - t_start:.1f} 秒\n")

    except Exception as e:
        print(f"\n❌ [{filename}] 下載中斷: {e}")
        raise

    return target_path

def run_download():
    print("=" * 72)
    print("⚡ [PULSAR-ASM] Google Gemma-2B-IT 免咒語公開鏡像串流下載器 ⚡")
    print("=" * 72)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    safetensors_dir = os.path.join(models_dir, "gemma_raw_safetensors")

    print(f"📁 暫存目錄: {safetensors_dir}")
    print(f"📁 目標目錄: {models_dir}\n")

    for filename, size in FILES_TO_DOWNLOAD:
        if filename.endswith(".safetensors"):
            download_file_with_resume(filename, size, safetensors_dir)
        else:
            download_file_with_resume(filename, size, models_dir)

    print("=" * 72)
    print("✅ 全部原廠權重與 Tokenizer 下載完成！")
    print("=" * 72)

if __name__ == "__main__":
    run_download()
