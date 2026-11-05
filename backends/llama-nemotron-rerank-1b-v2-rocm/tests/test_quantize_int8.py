"""quantize_int8 單元測試，不需要 GPU，也不需要下載模型。"""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quantize_int8 import WEIGHTS_FILE, convert, quantization_config, quantize_weight


def test_quantize_weight_uses_one_scale_per_row():
    weight = torch.tensor([[1.0, -0.5], [0.0, 254.0]], dtype=torch.bfloat16)

    quantized, scale = quantize_weight(weight)

    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 1)
    assert quantized[0].tolist() == [127, -64]
    assert quantized[1].tolist() == [0, 127]
    assert scale[1].item() == pytest.approx(2.0)


def test_quantize_weight_error_is_within_half_a_step():
    weight = torch.randn(16, 64, dtype=torch.float32)

    quantized, scale = quantize_weight(weight)

    restored = quantized.to(torch.float32) * scale
    assert ((restored - weight).abs() <= scale / 2 + 1e-6).all()


def test_quantize_weight_handles_all_zero_row():
    quantized, scale = quantize_weight(torch.zeros(1, 4))

    assert quantized.tolist() == [[0, 0, 0, 0]]
    assert scale.item() == 1.0


def test_config_matches_vllm_dynamic_token_scheme():
    group = quantization_config()["config_groups"]["group_0"]

    assert group["weights"]["strategy"] == "channel"
    assert group["weights"]["symmetric"] is True
    assert group["weights"]["dynamic"] is False
    assert group["input_activations"]["strategy"] == "token"
    assert group["input_activations"]["dynamic"] is True


def test_convert_only_quantizes_decoder_linear_layers(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama_bidirec"}))
    (source / "tokenizer.json").write_text("{}")
    (source / "pytorch_model.bin").write_text("duplicate weights")
    save_file(
        {
            "model.embed_tokens.weight": torch.ones(4, 2, dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
            "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
            "model.layers.0.input_layernorm.weight": torch.ones(2, dtype=torch.bfloat16),
            "score.weight": torch.ones(1, 2, dtype=torch.bfloat16),
        },
        source / WEIGHTS_FILE,
    )
    output = tmp_path / "output"

    quantized_count, _ = convert(source, output)

    assert quantized_count == 2
    with safe_open(output / WEIGHTS_FILE, framework="pt") as weights:
        assert weights.get_tensor("model.layers.0.self_attn.q_proj.weight").dtype == torch.int8
        assert weights.get_tensor("model.layers.0.self_attn.q_proj.weight_scale").shape == (2, 1)
        assert weights.get_tensor("model.embed_tokens.weight").dtype == torch.bfloat16
        assert weights.get_tensor("model.layers.0.input_layernorm.weight").dtype == torch.bfloat16
        assert weights.get_tensor("score.weight").dtype == torch.bfloat16
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "llama_bidirec"
    assert config["quantization_config"]["format"] == "int-quantized"
    assert (output / "tokenizer.json").exists()
    assert not (output / "pytorch_model.bin").exists()
    assert not output.with_name("output.tmp").exists()
