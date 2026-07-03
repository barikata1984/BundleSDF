# コードレビュー結果 (2026-07-03)

feat/ros-one-online のコードベース全体レビュー結果. 参照・更新される文書として扱う (修正したら該当行の状態を更新する). 手法: 8 領域の finder が候補 39 件を抽出し, 7 領域の verifier が独立検証 (CONFIRMED / PLAUSIBLE / REFUTED). 生存 25 件 (CONFIRMED 17 / PLAUSIBLE 8), 棄却 8 件.

対象は upstream BundleSDF 由来のコードを含む全体であり, 本フォークの移植作業由来は #9 (dockerfile 手編集) と #10 (新規 SAM3 ノード) の 2 件のみ. 他は upstream から継承したバグ.

## 上位 10 件 (重大度順)

| # | 場所 | 判定 | 状態 | 内容 | 失敗シナリオ |
|---|---|---|---|---|---|
| 1 | `bundlesdf.py:606` | CONFIRMED | **修正済み** (d403e4c) | 参照フレーム再探索経路に `pdb.set_trace()` 残置 | マッチ数低下 (高速運動・遮蔽) で非対話実行が stdin 待ちで永久停止 |
| 2 | `BundleTrack/src/Bundler.cpp:1308` | CONFIRMED | 未修正 | runNerf が zmq 応答をサイズ未検証で `frames.size()*16` float memcpy | 短い応答でバッファ外読み → クラッシュ/姿勢破損 |
| 3 | `BundleTrack/src/FeatureManager.cpp:1661` | CONFIRMED | 未修正 | `confs_gpu` だけ cudaFree 漏れ (解放ループ 1703-1711 に含まれず) | 長時間トラッキングで GPU メモリリーク → CUDA OOM. VRAM 単調増加の一因の可能性 |
| 4 | `run_custom.py:84` | CONFIRMED | 未修正 | `use_segmenter=1` 経路のみマスク未リサイズ (reader は shorter_side=480 に縮小済み) | 解像度不一致のまま C++ へ → 前景マスク silent corruption / 範囲外アクセス. **ベンチは use_segmenter=0 で回避** |
| 5 | `bundlesdf.py:688` | CONFIRMED | 未修正 | 空マスク時の `np.percentile(空配列)` ガードなし | 完全遮蔽・フレームアウトで実行全体がクラッシュ |
| 6 | `loftr_wrapper.py:116` | PLAUSIBLE | 未修正 | マッチ 0 件で `mconf.min()/max()` (ログ行) が空配列 reduction | テクスチャ欠乏ペアで `predict()` ごと ValueError |
| 7 | `bundlesdf.py:727` | CONFIRMED | 未修正 | NeRF 同期待ちが子プロセスの死活を未確認 | 子が CUDA OOM 死 → 親が無限待機. `sync_max_delay` 変更後も残存 |
| 8 | `BundleTrack/src/Bundler.cpp:897` | CONFIRMED | 未修正 | 対応点ゼロで FAIL を立てるが return せず, ゼロ対応点のまま最適化続行し姿勢を無条件書き戻し (953-957) | global_corres 空のフレームで無意味な姿勢に上書き → 後続へ伝播 |
| 9 | `docker/ros-one.dockerfile:38` | CONFIRMED | **修正済み** (cc36ede) | 手編集の `-DCUDA_ARC_BIN` typo + `BUILD_LIST` 削除 | cmake が未知 -D を無視 → 全アーキ×全モジュールのフルビルドに静かに退行. ※疑われた行継続破壊は実 docker build で再現せず (Docker はコメント行を結合前に除去) |
| 10 | `ros/sam3_segmenter/scripts/sam3_segmenter_node.py:38` | CONFIRMED | 未修正 (別途 video_storage_device は Tier0 で対応) | `init_video_session` に dtype 未指定 (モデルは bf16) | セッションが fp32 のまま dtype 不一致 or bf16 効果無効 |

## 圏外の生存候補 (10 件制限でカット, 修正価値あり)

| 場所 | 判定 | 内容 |
|---|---|---|
| `run_ho3d.py:95` | CONFIRMED | `on_finish()` 未呼び出しで nerf ワーカーが join されず終了時ハング |
| `bundlesdf.py:673` | CONFIRMED | GUI 起動失敗 (ヘッドレス) 時に `started` 待ちで永久ループ. オンライン運用で該当 |
| `BundleTrack/src/Bundler.cpp:305` | CONFIRMED | `min_trans`/`trans_diff` を計算するが判定に未使用 (並進チェック死蔵) |
| `BundleTrack/src/FeatureManager.cpp:90/202` | CONFIRMED | ROI 幅の +1 (inclusive) 規約が 2 関数間で不整合 → 座標に恒常ずれの可能性 |
| `BundleTrack/CMakeLists.txt:70` | CONFIRMED | `find_package(PCL 1.8)` の下限宣言が `pcl::make_shared` (要 1.10+) と矛盾 |
| `BundleTrack/src/Frame.h:35` | CONFIRMED | serialize() 実体のない boost::serialization friend 宣言の残骸 |
| `nerf_runner.py:1304` | PLAUSIBLE | 法線計算の `grad_outputs=zeros` で normals 常時ゼロ (現状デッドパス, get_normals=True の呼出なし) |
| `nerf_runner.py:735` | PLAUSIBLE | `eikonal_weight>0` 設定時に `extras['normals']` で KeyError (既定 0) |
| `nerf_runner.py:639` | PLAUSIBLE | `get_gradients()` の UnboundLocalError / 属性誤り (呼び出し箇所なしのデッドコード) |
| `mycuda/common.cu:71,92` | PLAUSIBLE | 境界ケースで `while(1){}` の GPU 無限スピン (クラッシュでなくハング) |
| `Utils.py:381` | PLAUSIBLE | `get_level_quantized_points` が `pyramids` を返すコピペミス (呼び出しなし) |
| `tool.py:51` | PLAUSIBLE | precomputed 経路 + mask None で `mask_file` 未定義 NameError |
| `build.sh:4` | PLAUSIBLE | torch パス解決失敗時のエラーハンドリングなし (空文字で続行) |
| `sam3_segmenter_node.py:49` | CONFIRMED | `buff_size=2^24` が 4K rgb8 (約 25MB) より小さい |
| `loftr_wrapper.py:80` | PLAUSIBLE | permute 後に `shape[-1]==3` を見るグレースケール判定 (USE_GRAY:true で休眠) |
| cleanup 系 4 件 | CONFIRMED/PLAUSIBLE | 二重 `no_grad` (loftr_wrapper) / `kpts_to_ray_ids` の重複 GPU→CPU 転送 / `run_nerf`/`run_global_nerf` の約 150 行重複 / 死んだ代入 `bundlesdf.py:590` |

## 検証で棄却された候補 (REFUTED)

| 候補 | 棄却理由 |
|---|---|
| dockerfile 行継続破壊でビルド失敗 | 実 docker build で再現せず. Docker はコメント行を結合前に除去 (bash 直接実行との差) |
| `Bundler.cpp:293` spdlog 引数不足クラッシュ | 同梱 spdlog 1.3.1 はエラーハンドラで捕捉, クラッシュしない |
| `Segmenter.__int__` typo | 本体が no-op のため観測可能な影響なし |
| gridencoder の空 `atomicAdd(Half)` | `grid.py:50` のガード (C 偶数時のみ half 化) で到達不能 |
| `Utils.py:460` ray_depths の CPU 生成 | 全消費側が `.cuda()` 済みで発火しない |
| `mycuda/common.cu:151` CHECK_INPUT 漏れ | 唯一の呼出元が `.long().contiguous()` 済みで影響なし |
| Utils.h の pcl::make_shared | 誤帰属 (Utils.h に該当なし, 実体は Frame.cpp) |

## 未修正分の対応方針 (次セッション以降)

オンライン ROS 運用で実際に踏む順は #1→#7 だが #1 は修正済み. 優先度は:

1. **#3 confs_gpu リーク**: 場所特定済み, 修正は cudaFree 1 行追加. VRAM 単調増加の切り分けと同時に着手可能.
2. **#2 zmq memcpy overread / #8 FAIL 非 return**: C++ の堅牢性. runNerf は現状 dead code (Python 側 run_nerf ワーカーを使用) なので #2 の実害は低いが, C++ runNerf を使う経路に戻すなら要修正.
3. **#5 percentile 空配列 / #6 mconf 空 / #7 nerf 子死活**: Python 側, いずれも数行のガード追加. オンライン長時間運用の安定性に効く.
4. cleanup 系と圏外バグは perf 改善 (PERF_plan.md) と合わせて判断.
