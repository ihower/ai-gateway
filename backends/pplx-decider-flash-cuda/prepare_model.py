"""把官方 checkpoint 整理成 vLLM 能直接載入的目錄。

vLLM 0.34 內建 Decision 3.0（D3ForSequenceClassification）：Qwen3.5 backbone + readout.safetensors +
decision_config.json，正好是 pplx-decider v1.3 的格式，只差 vLLM 只接受 decision_config 的
format_version 1（v1.3 為 2，其餘用到的欄位相同）。這裡建立一個目錄，檔案以 symlink 指回原始
checkpoint，只另存一份改了 format_version 的 decision_config.json（原檔保留為
decision_config.orig.json）。

用法：
  uv run python prepare_model.py --src /mnt/data/models/pplx/pplx-decider-flash-v1.3 \
      --dest /mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 權重以外 vLLM 與 protocol 需要的檔案
AUX_FILES = (
    "config.json",
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "joint_config.json",
    "LICENSE",
    "NOTICE",
)
# vLLM 只讀 format_version 1；v1.3（format_version 2）用到的欄位 codes / token_ids / temperature /
# attention_mode / pooling 與 1 相同
VLLM_DECISION_FORMAT = 1


def write_decision_config(src: Path, dest: Path) -> None:
    config = json.loads((src / "decision_config.json").read_text())
    if config.get("attention_mode") != "noncausal_full_attention" or config.get("pooling", "last") != "last":
        raise SystemExit("預期 noncausal_full_attention + last pooling 的 checkpoint")
    original = dest / "decision_config.orig.json"
    original.unlink(missing_ok=True)
    original.symlink_to(src / "decision_config.json")
    config["format_version"] = VLLM_DECISION_FORMAT
    (dest / "decision_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n")


def link(src: Path, dest: Path, names) -> None:
    for name in names:
        target = dest / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(src / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="官方 checkpoint 目錄")
    parser.add_argument("--dest", type=Path, required=True, help="輸出目錄")
    args = parser.parse_args()
    src, dest = args.src.resolve(), args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    link(src, dest, [*AUX_FILES, "model.safetensors.index.json", *shards])
    write_decision_config(src, dest)
    print(f"已建立 {dest}")


if __name__ == "__main__":
    main()
