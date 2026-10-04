# ==============================================================================
# Project PULSAR-ASM | Tool: convert_gemma_safetensors.py
# ------------------------------------------------------------------------------
# High-Speed Zero-PyTorch Safetensors to Flat FP16 Raw Material Converter
# Converts Google Gemma-2B-IT Safetensors weights into contiguous AI-DOS format:
#   pulsar_asm/models/gemma2b_fp16.bin (5,012,496,384 Bytes)
#
# Memory Layout:
#   For Layer L = 0 .. 17 (Stride = 220,217,344 Bytes):
#     - rms_att_w   (2048 floats FP32 = 8,192 B)
#     - rms_ffn_w   (2048 floats FP32 = 8,192 B)
#     - w_q         (2048 x 2048 FP16 = 8,388,608 B)
#     - w_k         (256 x 2048 FP16  = 1,048,576 B)
#     - w_v         (256 x 2048 FP16  = 1,048,576 B)
#     - w_o         (2048 x 2048 FP16 = 8,388,608 B)
#     - w_gate      (16384 x 2048 FP16 = 67,108,864 B)
#     - w_up        (16384 x 2048 FP16 = 67,108,864 B)
#     - w_down      (2048 x 16384 FP16 = 67,108,864 B)
#   Followed by:
#     - final_norm_w (2048 floats FP32 = 8,192 B)
#     - token_emb    (256000 x 2048 FP16 = 1,048,576,000 B)
# ==============================================================================

import glob
import json
import os
import struct
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class SafetensorsIndex:
    def __init__(self, directory):
        self.directory = directory
        self.tensor_map = {}  # name -> (file_path, base_offset, dtype, shape, offsets)
        self._build_index()

    def _build_index(self):
        files = sorted(glob.glob(os.path.join(self.directory, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {self.directory}")

        for file_path in files:
            with open(file_path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header_json = json.loads(f.read(header_len).decode("utf-8"))
                base_offset = 8 + header_len
                for name, info in header_json.items():
                    if name == "__metadata__":
                        continue
                    self.tensor_map[name] = (
                        file_path,
                        base_offset,
                        info["dtype"],
                        info["shape"],
                        info["data_offsets"],
                    )
        print(f"📦 已建立 Safetensors 張量索引: 涵蓋 {len(self.tensor_map)} 組張量 (共 {len(files)} 個檔案)")

    def write_tensor_to_file(self, out_file, name, target_dtype="F16"):
        if name not in self.tensor_map:
            raise KeyError(f"Tensor {name} not found in safetensors!")

        file_path, base_offset, src_dtype, shape, (start, end) = self.tensor_map[name]
        total_bytes = end - start
        chunk_bytes = 4 * 1024 * 1024  # 4 MB chunk for low memory footprint

        with open(file_path, "rb") as f:
            f.seek(base_offset + start)
            bytes_read = 0
            while bytes_read < total_bytes:
                read_size = min(chunk_bytes, total_bytes - bytes_read)
                raw = f.read(read_size)
                bytes_read += read_size

                if src_dtype == "BF16":
                    u16 = np.frombuffer(raw, dtype=np.uint16)
                    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                elif src_dtype == "F16":
                    f16 = np.frombuffer(raw, dtype=np.float16)
                    f32 = f16.astype(np.float32)
                elif src_dtype == "F32":
                    f32 = np.frombuffer(raw, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported source dtype: {src_dtype}")

                if target_dtype == "F16":
                    out_file.write(f32.astype(np.float16).tobytes())
                elif target_dtype == "F32":
                    out_file.write(f32.astype(np.float32).tobytes())

def convert_safetensors_to_pulsar(safetensors_dir, output_bin_path):
    print("=" * 72)
    print("⚡ [PULSAR-ASM] Google Gemma-2B-IT Safetensors 全精度轉碼器 (流式低內存版) ⚡")
    print("=" * 72)

    idx = SafetensorsIndex(safetensors_dir)
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)

    expected_layer_bytes = 220217344
    n_layers = 18
    total_written = 0

    print(f"\n🚀 開始封裝 18 層 Gemma-2B 原裝全精度權重至: {output_bin_path}...")

    with open(output_bin_path, "wb") as out:
        for L in range(n_layers):
            layer_start = out.tell()
            print(f"   [Layer {L:02d}/18] 萃取中...", end="", flush=True)

            idx.write_tensor_to_file(out, f"model.layers.{L}.input_layernorm.weight", target_dtype="F32")
            idx.write_tensor_to_file(out, f"model.layers.{L}.post_attention_layernorm.weight", target_dtype="F32")
            idx.write_tensor_to_file(out, f"model.layers.{L}.self_attn.q_proj.weight", target_dtype="F16")
            idx.write_tensor_to_file(out, f"model.layers.{L}.self_attn.k_proj.weight", target_dtype="F16")
            idx.write_tensor_to_file(out, f"model.layers.{L}.self_attn.v_proj.weight", target_dtype="F16")
            idx.write_tensor_to_file(out, f"model.layers.{L}.self_attn.o_proj.weight", target_dtype="F16")
            idx.write_tensor_to_file(out, f"model.layers.{L}.mlp.gate_proj.weight", target_dtype="F16")
            idx.write_tensor_to_file(out, f"model.layers.{L}.mlp.up_proj.weight", target_dtype="F16")
            idx.write_tensor_to_file(out, f"model.layers.{L}.mlp.down_proj.weight", target_dtype="F16")

            layer_end = out.tell()
            layer_len = layer_end - layer_start
            assert layer_len == expected_layer_bytes, f"Layer {L} length mismatch: {layer_len} vs {expected_layer_bytes}"
            print(f" 完成 ({layer_len / (1024*1024):.2f} MB)")

        # Final Norm
        print("   [Final Norm] 萃取中...", end="", flush=True)
        idx.write_tensor_to_file(out, "model.norm.weight", target_dtype="F32")
        print(" 完成")

        # Token Embedding (tied with LM head)
        print("   [Token Embedding (256K)] 萃取中...", end="", flush=True)
        idx.write_tensor_to_file(out, "model.embed_tokens.weight", target_dtype="F16")
        print(" 完成 (1000.00 MB)")

        total_written = out.tell()

    expected_total = 18 * expected_layer_bytes + 2048 * 4 + 256000 * 2048 * 2
    print(f"\n🎯 [出廠檢驗] 轉出檔案大小: {total_written:,} Bytes ({total_written / (1024**3):.3f} GB)")
    assert total_written == expected_total, f"Total size mismatch: {total_written} vs {expected_total}"
    print("=" * 72)
    print("✅ [轉碼合格] gemma2b_fp16.bin 封裝成功，完全對齊 AI-DOS 純組合語言引擎規格！")
    print("=" * 72)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_gemma_safetensors.py <safetensors_dir> <output_bin_path>")
        sys.exit(1)
    convert_safetensors_to_pulsar(sys.argv[1], sys.argv[2])
