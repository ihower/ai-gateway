"""vLLM platform plugin：強制使用 ROCm。

這台機器同時有 NVIDIA（RTX 5080 / 5070 Ti）和 AMD（Radeon 8060S，gfx1151）GPU。
vLLM 內建的偵測會同時啟用 cuda（pynvml 看得到 NVIDIA）和 rocm，然後直接報錯：
"Only one platform plugin can be activated"。內建偵測沒有環境變數可以關掉，
但 out-of-tree platform plugin 的優先權高於內建偵測，所以註冊這個 plugin
（pyproject.toml 的 vllm.platform_plugins entry point）把平台固定為 ROCm。
"""


def force_rocm() -> str | None:
    return "vllm.platforms.rocm.RocmPlatform"
