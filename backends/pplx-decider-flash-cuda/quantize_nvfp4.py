"""以 NVIDIA ModelOpt 把 pplx-decider-flash-v1.3 量化成 NVFP4（W4A4），輸出 vLLM 可直接載入的目錄。

- 校準時 full attention 為 noncausal（與官方 runtime、vLLM serving 相同），GDN（linear attention）維持 causal。
  只用 AutoModel 載入會是 causal，activation scale 會校準在錯的分佈上。
- 校準資料：tests/fixtures/calibration-requests.json（fxoz/pplx-decider-flash-v1.3-GGUF 的校準集）
  加上 tests/synthetic.py 的 calib split，依官方 protocol v4 編碼成 separate mode 的 sequences。
  評估用的 tests/fixtures/reference.json 與 synthetic eval split 不參與校準。
- ModelOpt 預設已排除 vision tower、GDN 的 conv1d / in_proj_a / in_proj_b；--keep 可再指定保留 BF16 的模組
  （例如 '*self_attn*'）。readout.safetensors 不經量化，原樣保留 BF16。

用法（需要 quantize dependency group：uv sync --group quantize）：
  CUDA_VISIBLE_DEVICES=1 uv run --group quantize python quantize_nvfp4.py \
      --src /mnt/data/models/pplx/pplx-decider-flash-v1.3 \
      --dest /mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-nvfp4 --keep '*self_attn*'
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from prepare_model import AUX_FILES, write_decision_config  # noqa: E402


def enable_noncausal_full_attention(text_model) -> None:
    """與官方 pplx_decider/model.py 相同：full attention 看整段（非 padding）sequence，linear attention 維持 causal。"""
    import torch
    from transformers.masking_utils import create_recurrent_attention_mask

    def mask_inputs(module, args, kwargs):
        if args:
            raise ValueError("The noncausal mask hook requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Noncausal classification does not support a KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if isinstance(padding, dict):
            return args, kwargs
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding
            ),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


def calibration_sequences(src: Path, max_length: int) -> list[list[int]]:
    import synthetic
    from decider_codec import DecisionCodec

    codec = DecisionCodec(src, max_length=max_length)
    reqs = json.loads((ROOT / "tests" / "fixtures" / "calibration-requests.json").read_text())
    reqs += synthetic.requests("calib")
    return [chunk.input_ids.tolist() for request in reqs for chunk in codec.plan(request)]


def recipe(name: str, keep: list[str]) -> dict[str, Any]:
    import modelopt.torch.quantization as mtq

    config = copy.deepcopy(getattr(mtq, name))
    config["quant_cfg"].extend({"quantizer_name": pattern, "enable": False} for pattern in keep)
    return config


def check_noncausal(model, ids) -> float:
    """改最後一個 token，第一個位置的輸出要跟著變（causal 時不會變）。"""
    import torch

    alt = ids.clone()
    alt[0, -1] = (alt[0, -1] + 1) % 1000 + 10
    with torch.no_grad():
        a = model(input_ids=ids, use_cache=False).last_hidden_state[0, 0].float()
        b = model(input_ids=alt, use_cache=False).last_hidden_state[0, 0].float()
    return float((a - b).abs().max())


#: quantization_config 的 targets 是所有 Linear，vLLM 多建的兩個模組必須排除，否則會被當成 NVFP4：
#: - `score`：vLLM 把 readout 當成 classification head（ReplicatedLinear），權重是 BF16 的 readout.safetensors
#: - `lm_head`：D3 繼承自 Qwen3_5ForConditionalGeneration，有個不會用到的 lm_head（與 embedding 綁定）；
#:   checkpoint 沒有它的 NVFP4 權重，不排除會停在 meta tensor 而載入失敗
VLLM_ONLY_MODULES = ("score", "lm_head")


def finalize_config(dest: Path) -> None:
    """architectures 維持 Qwen3_5Model（vLLM 才會辨識成 D3），並把 vLLM 多建的模組加進不量化清單。"""
    config = json.loads((dest / "config.json").read_text())
    config["architectures"] = ["Qwen3_5Model"]
    ignore = config["quantization_config"].setdefault("ignore", [])
    ignore.extend(name for name in VLLM_ONLY_MODULES if name not in ignore)
    (dest / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    quant = dest / "hf_quant_config.json"
    if quant.exists():
        data = json.loads(quant.read_text())
        exclude = data["quantization"].setdefault("exclude_modules", [])
        exclude.extend(name for name in VLLM_ONLY_MODULES if name not in exclude)
        quant.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="官方 BF16 checkpoint 目錄")
    parser.add_argument("--dest", type=Path, required=True, help="輸出目錄（會覆寫）")
    parser.add_argument("--recipe", default="NVFP4_DEFAULT_CFG", help="modelopt.torch.quantization 的 recipe 名稱")
    parser.add_argument("--keep", action="append", default=[], help="保留 BF16 的模組 glob，可重複指定")
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()

    import modelopt.torch.quantization as mtq
    import torch
    from modelopt.torch.export import export_hf_checkpoint
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    src, dest = args.src.resolve(), args.dest.resolve()
    sequences = calibration_sequences(src, args.max_length)
    print(f"calibration: {len(sequences)} sequences, {sum(map(len, sequences))} tokens", flush=True)

    torch.backends.cuda.enable_cudnn_sdp(False)  # 官方 runtime 的 SDPA kernel 選擇
    model = Qwen3_5Model.from_pretrained(str(src), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.eval().requires_grad_(False)
    enable_noncausal_full_attention(model.language_model)
    batches = [torch.tensor([ids], device="cuda") for ids in sequences]
    diff = check_noncausal(model, batches[0])
    print(f"noncausal check: first-token max |Δ| = {diff:.4g}", flush=True)
    if diff <= 1e-3:
        raise SystemExit("full attention 沒有變成 noncausal，停止")

    def forward_loop(m) -> None:
        with torch.no_grad():
            for batch in batches:
                m(input_ids=batch, use_cache=False)

    config = recipe(args.recipe, args.keep)
    started = time.time()
    mtq.quantize(model, config, forward_loop=forward_loop)
    print(f"quantize: {time.time() - started:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    mtq.print_quant_summary(model)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    export_hf_checkpoint(model, export_dir=str(dest))

    finalize_config(dest)
    for name in AUX_FILES:
        if name != "config.json":
            shutil.copy2(src / name, dest / name)
    write_decision_config(src, dest)
    (dest / "decision_config.orig.json").unlink()
    shutil.copy2(src / "decision_config.json", dest / "decision_config.orig.json")
    (dest / "quantization-recipe.json").write_text(json.dumps({
        "source": str(src),
        "recipe": args.recipe,
        "keep_bf16": args.keep,
        "calibration_sequences": len(sequences),
        "calibration_attention": "noncausal_full_attention",
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"exported -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
