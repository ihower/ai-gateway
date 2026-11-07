#!/bin/sh
# 編譯 release 版並複製到根目錄的 ./ai-gateway（此檔進 git，部署端 git pull 後可直接使用，不需要 Rust）。
set -eu
cd "$(dirname "$0")"
cargo build --release
# 先複製再 rename，gateway 執行中也能替換（舊 process 繼續用原本的檔案）
cp target/release/ai-gateway ./ai-gateway.new
mv -f ./ai-gateway.new ./ai-gateway
# 顯示需要的最低 glibc 版本，部署端的 glibc 不能比這個舊
glibc=$(objdump -T ai-gateway 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1 || true)
echo "built ./ai-gateway ($(du -h ai-gateway | cut -f1), requires ${glibc:-unknown})"
