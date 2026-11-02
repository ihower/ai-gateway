# Benchmark：NVIDIA vs ROCm vs CPU

2026-11-02 在同一台機器上，比較三個 voyage-4-nano embedding server 的速度。三個都透過相同的 OpenAI 相容 `/v1/embeddings` API 呼叫。

## 受測環境

| server | port | 硬體 / 執行環境 |
|---|---|---|
| NVIDIA | 12352 | RTX 5070 Ti 16GB，vLLM 0.16.0（CUDA），bf16，`--enforce-eager` |
| ROCm | 12353 | Radeon 8060S（Strix Halo，gfx1151，統一記憶體），vLLM 0.19.1+rocm7.13.0，bf16，`--enforce-eager`，`ROCM_ATTN` backend |
| CPU | 12359 | CPU fallback server |

評測前三台都先跑過 `test_client.py`，全部 ALL PASS。檢索分數一致（火星文件約 0.651 / 0.567，金星文件約 0.403），確認比較的是同樣正確的輸出。

## 結果

| server | 單筆 query 延遲（p50） | 長文件約 2k token（p50） | 批次吞吐量 docs/s | tokens/s |
|---|---|---|---|---|
| NVIDIA 5070 Ti | 7.4 ms | 27.8 ms | 706 | 166k |
| ROCm 8060S | 7.9 ms | 64.1 ms | 132 | 31k |
| CPU | 22.0 ms | 751 ms | 34.5 | 8.1k |

相對倍數：

| | 長文件延遲 | 吞吐量 |
|---|---|---|
| 5070 Ti 對 8060S | 快 2.3 倍 | 5.4 倍 |
| 8060S 對 CPU | 快 12 倍 | 3.8 倍（CPU 的測法不同，見下方） |

## 解讀

- **短 query 三台都快**：兩張 GPU 都約 8 ms，CPU 22 ms。這時主要花在 HTTP 和排程上，GPU 算力差別看不出來，即時搜尋用哪台都可以。
- **算力差在長文件和批次**：5070 Ti 比 8060S 快 2.3～5.4 倍。8060S 的記憶體頻寬比獨立顯卡低，ROCm 上的 kernel 也比較不成熟。
- **兩張 GPU 都跑 eager 模式**（`VoyageQwen3BidirectionalEmbedModel` 的 torch.compile bug），所以兩邊條件相同。vLLM 修好這個 bug 之後，兩邊應該都會再快一些。
- **建議**：大量建索引用 5070 Ti。8060S 適合日常查詢，或在 NVIDIA 卡被其他工作佔用時當備援。

## 測試方法

測試文字是中英混合的火星描述，重複組成指定長度。每項測試前都先送一次 warmup 請求。

| 項目 | GPU | CPU |
|---|---|---|
| 單筆 query 延遲 | 短 query（`input_type=query`）循序送 30 次，取中位數 | 5 次 |
| 長文件延遲 | 單份約 2k token 文件，循序送 10 次，取中位數 | 2 次 |
| 批次吞吐量 | 每個請求 32 份文件（每份約 235 token），4 個併發，共 16 個請求 | 每批 16 份，不併發，共 2 個請求 |

注意事項：

- CPU 太慢，吞吐量測試的批次大小、併發數和請求數都縮小了，和 GPU 不是同一個測法，數字只能當參考。
- 每台只跑一輪，差距在 ±10% 內的數字不要過度解讀。
- 三台 server 同時在線，評測時是依序測，不是同時測，彼此不會搶 GPU；但 8060S 是統一記憶體，系統 RAM 的負載可能會影響它。
