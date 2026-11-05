#!/usr/bin/env python3
"""把 BF16 權重轉成 vLLM 可讀的 INT8 checkpoint（compressed-tensors int-quantized）。

對應 compressed-tensors 的 INT8_W8A8 preset：
- 權重：逐 channel、對稱 INT8，轉換時算好 scale
- 輸入：推論時由 vLLM 動態逐 token 量化，所以不需要校準資料

只量化 decoder layer 的線性層。embedding、norm 和 score 分類頭維持 BF16。

用法：
    uv run python quantize_int8.py
    uv run python quantize_int8.py --output models/my-int8 --force
"""

import argparse
import fnmatch
import json
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models" / "llama-nemotron-rerank-1b-v2-int8"
WEIGHTS_FILE = "model.safetensors"
# vLLM 載入需要的檔案。其他像 pytorch_model.bin 這種重複權重不要下載，也不要複製
SUPPORT_FILES = ("config.json", "tokenizer*", "special_tokens_map.json", "llama_bidirectional_model.py")
LINEAR_WEIGHT = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"
)
INT8_MAX = 127


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """回傳 (int8 權重, float32 scale)。channel 是輸出維度，也就是矩陣的列。"""
    values = weight.to(torch.float32)
    amax = values.abs().amax(dim=1, keepdim=True)
    # 全零的列 scale 用 1.0，避免除以零
    scale = torch.where(amax > 0, amax / INT8_MAX, torch.ones_like(amax))
    quantized = (values / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized, scale


def quantization_config() -> dict:
    def args(strategy: str, dynamic: bool) -> dict:
        return {
            "num_bits": 8,
            "type": "int",
            "strategy": strategy,
            "symmetric": True,
            "dynamic": dynamic,
        }

    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "ignore": ["score"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": args("channel", dynamic=False),
                "input_activations": args("token", dynamic=True),
            }
        },
    }


def convert(source: Path, output: Path) -> tuple[int, int]:
    tensors: dict[str, torch.Tensor] = {}
    quantized_count = 0
    with safe_open(source / WEIGHTS_FILE, framework="pt") as weights:
        for name in weights.keys():
            tensor = weights.get_tensor(name)
            if LINEAR_WEIGHT.match(name):
                tensors[name], tensors[f"{name[: -len('.weight')]}.weight_scale"] = quantize_weight(tensor)
                quantized_count += 1
            else:
                tensors[name] = tensor

    # 先寫到暫存目錄，中斷時不會留下看似完整的 checkpoint
    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in SUPPORT_FILES):
            shutil.copy(path, temporary / path.name)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = quantization_config()
    (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(tensors, temporary / WEIGHTS_FILE, metadata={"format": "pt"})

    if output.exists():
        shutil.rmtree(output)
    temporary.rename(output)
    return quantized_count, (output / WEIGHTS_FILE).stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo 或本地目錄")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="覆寫已存在的輸出目錄")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} 已存在，要覆寫請加 --force")

    model_path = Path(args.model)
    source = model_path if model_path.is_dir() else Path(snapshot_download(args.model, allow_patterns=[WEIGHTS_FILE, *SUPPORT_FILES]))
    quantized_count, size = convert(source, args.output)
    print(f"已量化 {quantized_count} 個線性層，權重 {size / 2**30:.2f} GiB：{args.output}")


if __name__ == "__main__":
    main()
