# 推論速度 改善計画 (2026-07-03)

オンライントラッキング (run_custom.py --mode run_video 相当) の推論速度を悪化させる実装の一覧と改善方針. 参照・更新される文書として扱う (対応したら状態を更新する). 手法: Python トラッキング経路 / C++ ホットパス / NeRF 並行学習 + SAM3 セグメンタの 3 領域を調査.

Tier 0 は即効 1 行修正で全て対応済み (コミット a7a3ec4). Tier 1 以降は変更量が大きく, ベンチ内訳分析 (perf CSV) で費用対効果を確定してから着手する.

## Tier 0: 1 行変更で即効 (全て対応済み)

| # | 場所 | 内容 | 状態 |
|---|---|---|---|
| 0-1 | `config.yml:102` | `sync_max_delay: 0→3`. 0 では NeRF 学習が実質フルシンクで, キーフレームごとにトラッキングが数秒〜十数秒停止していた | 対応済み |
| 0-2 | `docker/ros-one.dockerfile` | `ENV OMP_NUM_THREADS=1` 削除. C++ の全 `#pragma omp parallel for` を単一スレッド化していた. ※Python 側 multiprocessing との干渉は A/B 未確認 (再ベンチで要検証) | 対応済み |
| 0-3 | `loftr_wrapper.py:134` | 毎 `predict()` の `torch.cuda.empty_cache()` 削除. アロケータ全解放で以降の CUDA 割り当てが cudaMalloc 再発行を強制していた | 対応済み |
| 0-4 | `sam3_segmenter_node.py:41` | `video_storage_device='cpu'→self.device`. 毎フレーム GPU→CPU→GPU の往復転送 (1008², ストリーミングでは無意味) | 対応済み |

## Tier 1: 構造的・効果大 (変更量あり, ベンチ後に着手)

| # | 場所 | 内容 | 改善案 |
|---|---|---|---|
| 1-1 | `BundleTrack/src/cuda/CUDAImageUtil.cu` | 毎フレーム最低 7 回の `cudaDeviceSynchronize()` (erode→gauss×2→convert→normals→edge filter→convert) がデプス処理パイプラインを逐次化 | 単一 stream で連続発行, 同期は CPU 読み出し直前の 1 回に集約 |
| 1-2 | `Frame.cpp:126-132` / `FeatureManager.cpp:1655` | 毎フレームの `cudaMalloc/cudaFree` (Frame の GPU バッファ + RANSAC ペアごと 7 本, ペア数の 2 乗で増加). cudaMalloc は暗黙の全体同期 | 解像度固定バッファのプール化 or `cudaMallocAsync` |

## Tier 2: 毎フレームの転送・コピー税

| # | 場所 | 内容 | 改善案 |
|---|---|---|---|
| 2-1 | `loftr_wrapper.py:80` | `.float()` が `.cuda()` より前で H2D 転送量 4 倍 (uint8 3.5MB → float32 13.8MB/回) | GPU 転送後に float 化 |
| 2-2 | `bundlesdf.py:511` | `np.array([np.array(img) for img in imgs])` の二重コピー | `np.stack(imgs)` |
| 2-3 | `bundlesdf.py:439,707` | Manager 経由でキーフレーム画像を pickle 二重転送 (クライアント→マネージャサーバ→クライアント) | `shared_memory` / Queue |
| 2-4 | `FeatureManager.cpp:2330` / `Frame.cpp:28-34` | zmq 送信の二重コピー / `Frame` の color×2 + depth×3 の `clone()` | ゼロコピー送信 / copy-on-write |

## Tier 3: スケールで効く・Tier 0 修正後に顕在化

| # | 場所 | 内容 | 改善案 |
|---|---|---|---|
| 3-1 | `Bundler.cpp:298,509` | キーフレーム全走査が上限なしで O(N²) 化, 長時間オンラインで単調悪化. ベンチで序盤 18.7fps → 蓄積で減速として観測 | 上限設定 or コメントアウト中の GPU 版 covisibility 復活 |
| 3-2 | プロセス構成 | MPS 未使用でトラッキング/NeRF の 2 プロセスがコンテキストスイッチ (50-100μs/回) で削り合う (0-1 修正後に顕在化) | `nvidia-cuda-mps-control` |
| 3-3 | 小物 | `astype(bool)` 冗長コピー (sam3 node), f-string ログ即時評価, `corres` 分割の線形走査, 二重 `no_grad` | 各 1 行 |

## ベンチマーク結果 (2026-07-03, ミルク動画 1932 フレーム, RTX 5090)

Tier 0 適用後の計測. `use_segmenter=0` (データ同梱マスク使用).

| backend | median | mean | p90 | wall-clock | >1s フレーム | 序盤 100f |
|---|---|---|---|---|---|---|
| eloftr | 162.1 ms | 352.1 ms | 227.2 ms | 11分49秒 | 107 | 53ms (18.7fps) |
| loftr | 177.7 ms | 362.1 ms | 241.1 ms | 12分08秒 | 104 | — |

- **姿勢整合性** (eloftr vs loftr, GT なしの相対比較): 並進差 median 0.16cm / p90 0.45cm, 回転差 median 1.33° / max 19.6° (低テクスチャ面での対称性曖昧が疑い). 絶対精度は HO3D + benchmark_ho3d.py で別途.
- **VRAM**: 17GB → 26GB へ単調増加 (OOM なし). キーフレーム蓄積が主因だが REVIEW_findings #3 の confs_gpu リークも寄与の可能性. 要切り分け.
- per-stage 内訳 CSV: `data/out_milk_{eloftr,loftr}/perf_main.csv` / `perf_nerf.csv` / `perf_nerf_train.csv` (BUNDLESDF_PROFILE=1). 定常フレームの支配項は `loftr_predict`, 裾は NeRF 学習・グローバル BA.

## EfficientLoFTR の速度差が小さかった理由 (median で 9%, 期待は 2.5 倍)

1. **マッチャーはフレーム予算の少数派 (Amdahl)**: median 162ms の中身はデプス前処理・点群/法線・マッチング・RANSAC・ポーズグラフ最適化・キーフレーム管理の合算. マッチングが仮に 40ms なら 2.5 倍でも稼ぎは約 24ms で, 観測差 15.6ms と整合. パイプラインはマッチャー律速ではない.
2. **入力レジームが論文と異なる**: `feature_corres.resize: 400` の 400×400 グレースケール (eloftr は 416 にパディング). 論文ベンチ (640 級以上) より小さく, RTX 5090 ではカーネル起動・同期オーバーヘッド律速に近づき FLOPs 差が時間差に転写されにくい. 400 は 8 で割り切れるため旧 LoFTR はパディング不要, eloftr だけ 416² (+8% 画素) のハンデ.
3. **full+mp 設定**: 精度優先の `full_default_cfg` + autocast を採用. 論文の最速数字は opt 設定 (skip_softmax + fp16matmul) 側. full ではマッチャー単体でも 2.5 倍は出ない. opt 設定への切替が追加の伸びしろ.
4. **mean/wall-clock は裾が支配**: 1 秒超の約 105 フレーム (NeRF 学習・グローバル BA) が総時間の相当部分を占め, マッチャーと無関係. wall-clock 差が 2.6% しかないのはこのため.

**含意**: 速度目的の次の一手はマッチャーではなく Tier 1 (C++ 同期・アロケーション) と Tier 3-1 (キーフレーム走査上限). EfficientLoFTR 化は速度では小勝ちだが精度整合 (並進サブ cm) は出ており, opt 設定の伸びしろも残る.

## プロセス構造 (調査済み事実, 計測の前提)

- 2 プロセス構成: (A) メイントラッキング (Python + my_cpp, 同一プロセス), (B) NeRF ワーカー (`multiprocessing.Process`, spawn). GUI プロセスはベンチ (use_gui=0) では起動しない.
- 通信は全て `multiprocessing.Manager` 経由 (p_dict / kf_to_nerf_list / Lock). **zmq 3 ポート (port/seg_port/nerf_port) はこの経路で未使用** (ソケット接続のみ, 送信なし).
- **C++ `Bundler::runNerf` はどこからも呼ばれない dead code** (Python 側 run_nerf ワーカーが実体). REVIEW_findings #2 の zmq memcpy はこの dead code 内にあり実害は低い.
- **C++ `Utils::Timer` は未インスタンス化**: `-DTIMER=1` に変えても出力なし. C++ 内訳計測には Timer 挿入が別途必要.
- C++ 呼び出し (optimizeGPU, runRansacMultiPairGPU, Frame 構築等) は Python の 1 行呼び出しとして境界が切れるため, C++ を触らずに塊単位の時間は Python 側計測 (perf_logger.py) で取れる. 内訳は取れない.
