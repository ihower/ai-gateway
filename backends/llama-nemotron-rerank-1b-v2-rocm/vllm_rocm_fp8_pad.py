"""vLLM general plugin：讓 ROCm 上的 `--quantization fp8_per_tensor` 和 NVIDIA 版（12361）算法一致。

vLLM 的 Fp8PerTensorOnlineLinearMethod 在兩個平台上的行為不同：

1. activation 量化方式
   - NVIDIA（有 CUTLASS）：權重 per-tensor、activation **逐 token** 動態量化（kFp8DynamicTokenSym）。
     每個 token 有自己的 scale，分數和同一批有哪些文件無關。
   - ROCm（沒有 CUTLASS）：activation 退回 **per-tensor**（kFp8DynamicTensorSym），整批 token 共用
     一個 scale，分數會隨同批文件改變，和 12361 的差距也較大。
   這裡在 ROCm 上改用 kFp8DynamicTokenSym，vLLM 會選 ChannelWiseTorchFP8ScaledMMLinearKernel：
   torch._scaled_mm 輸出 fp32，再乘上 token scale 和權重 scale，和 CUTLASS 的 epilogue 相同。

2. 輸出維度對齊
   reranker 的 `score` 層是 2048 → 1。hipBLASLt 的 torch._scaled_mm 要求 N 為 16 的倍數，
   會報 `mat2 shape (2048x1) must be divisible by 16`。CUTLASS kernel 會把權重補零到 16 的倍數，
   所以 NVIDIA 版的 `score` 層也是 FP8。這裡同樣在呼叫前把 B（[K, N]）補零，算完切回原寬度。

由 VLLM_PLUGINS 控制是否載入（start.sh 預設載入；不加 --quantization 時不影響任何東西）。
vLLM 會在每個 process 各載入一次，所以 patch 要能重複執行。
"""

_ALIGNMENT = 16


def _pad_n(original):
    import torch

    def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
        n = B.shape[1]
        pad_n = (-n) % _ALIGNMENT
        if pad_n == 0:
            return original(
                self, A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs, bias=bias,
                output_shape=output_shape,
            )
        # B 是 column-major 的 [K, N]：在 [N, K] 上補列再轉置回去，維持 column-major
        B = torch.nn.functional.pad(B.t(), (0, 0, 0, pad_n)).t()
        if Bs.numel() > 1:  # 逐 channel 的權重 scale，形狀 (N,) 或 (N, 1)：補 1
            flat = torch.nn.functional.pad(Bs.reshape(-1), (0, pad_n), value=1.0)
            Bs = flat.view(-1, 1) if Bs.dim() == 2 else flat
        if bias is not None:
            bias = torch.nn.functional.pad(bias, (0, pad_n))
        padded_shape = [*output_shape[:-1], n + pad_n]
        out = original(
            self, A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs, bias=bias,
            output_shape=padded_shape,
        )
        return out[..., :n].contiguous()

    apply_scaled_mm._pads_n = True
    return apply_scaled_mm


def rocm_fp8_pad() -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return

    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
        ROCmFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
        kFp8DynamicTokenSym,
    )

    for kernel in (ChannelWiseTorchFP8ScaledMMLinearKernel, ROCmFP8ScaledMMLinearKernel):
        if not getattr(kernel.apply_scaled_mm, "_pads_n", False):
            kernel.apply_scaled_mm = _pad_n(kernel.apply_scaled_mm)

    init = Fp8PerTensorOnlineLinearMethod.__init__
    if getattr(init, "_per_token", False):
        return

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.activation_quant_key = kFp8DynamicTokenSym

    __init__._per_token = True
    Fp8PerTensorOnlineLinearMethod.__init__ = __init__
