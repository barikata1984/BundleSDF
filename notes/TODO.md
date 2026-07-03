# TODO

## 完了

- [x] コンテナ定義ファイル作成 (`docker/ros-one.dockerfile`, `docker/segmenter.dockerfile`)
- [x] 依存ベンダリング (`pytorch3d_transforms/`, `BundleTrack/EfficientLoFTR/` + `loftr_wrapper.py` 切替)
- [x] PCL 1.12 対応 (`Utils.h/cpp`, `Frame.cpp` の shared_ptr 化) / rgbd include 削除 / CMakeLists sm_120 分岐 / build.sh 修正

## 完了 (2026-07-03 追加)

- [x] ミルクデモ動画のダウンロードと `data/2022-11-18-15-10-24_milk/` への展開 (rgb/depth/masks 1932枚ずつ, 構造検証済み)
- [x] 重み配置 (eloftr_outdoor.ckpt=Google Drive 193MB, LoFTR outdoor_ds.ckpt 46MB). SAM3 は対象外 (別issue)
- [x] ベンチスクリプト作成: `scripts/bench_milk.py` (フレーム毎 track() 計測), `scripts/compare_poses.py` (eloftr/loftr 姿勢軌跡比較), `scripts/bench_milk.sh` (オーケストレーション). 構文チェックのみ, 未実行
- [x] コンテナ内実行時検証: `bash build.sh` のビルドエラー4件 (mycuda pyproject.toml の torch pin, Utils.h テンプレート推論, FeatureManager.cpp の pcl/common/geometry.h include 追加, loftr_wrapper.py の torch.load weights_only 対応) を修正しビルド成功を確認
- [x] 上記ビルド修正後: my_cpp import 成功, eloftr/loftr 実マッチング動作確認, `scripts/bench_milk.sh` 実行 (ミルクベンチ全1932フレーム, eloftr/loftr 双方エラーゼロで完走)

## 未完了

- [ ] コンテナ (`ros-one.dockerfile`, SAM3 統合済み単一イメージ) の GHCR push (`segmenter.dockerfile` は py3.10 統合により廃止)
- [ ] SAM3 重み配置 (HF gated リポジトリ要承認申請)
- [x] EfficientLoFTR vs 旧 LoFTR の ho3d データセットでの ADD/ADD-S ベンチマーク比較 (2026-07-03). `BUNDLESDF_MATCHER=loftr PYTHONPATH=/workspace/mycuda python3 run_ho3d.py --video_dirs data/HO3D_v3/evaluation/SM1 --out_dir data/bench_results/ho3d_ours_loftr` で SM1 (mustard bottle, 898 フレーム) を loftr バックエンドで完走させ (898/898 フレームで `ob_in_cam/*.txt` 出力, gridencoder エラー・Traceback・CUDA error・Killed いずれもゼロ, 壁時計約 364.6 秒), config.yml/コードは無変更で環境変数 `PYTHONPATH=/workspace/mycuda` のみで gridencoder 問題 (`notes/ISSUES.md` 記載の既知の環境問題) を回避した. eloftr の既存結果は上書きせず別ディレクトリ (`data/bench_results/ho3d_ours_loftr/`, `data/bench_results/ho3d_log_loftr/`) に分離して保存. `benchmark_ho3d.py` で生 pkl から再計算し検証した精度比較 (n=895, 両者共通で 898 フレーム中 3 フレームは評価対象外, 同一 warm-start 設定・同一 seed=0) は次のとおり.

  | 指標 | eloftr (既定, 現行) | loftr (旧) | 差 |
  |---|---|---|---|
  | ADD mean (cm) ↓ | 2.20 (2.1954) | 2.77 (2.7671) | +0.57cm (loftr 悪化, +26%) |
  | ADD-S mean (cm) ↓ | 0.98 (0.9833) | 1.21 (1.2083) | +0.23cm (loftr 悪化, +23%) |
  | ADD_AUC (%) ↑ | 78.10 (78.099) | 72.39 (72.389) | -5.71pt (loftr 悪化) |
  | ADDS_AUC (%) ↑ | 90.18 (90.184) | 87.94 (87.939) | -2.24pt (loftr 悪化) |
  | chamfer (cm) | 0.52 (0.518) | 0.525 | +0.005cm (実質同等, NeRF 由来でバックエンド非依存のため妥当) |

  姿勢精度 (ADD/ADD-S/AUC) の 4 指標すべてで eloftr が明確に優位 (ADD_AUC で 5.7 ポイント, ADD 平均で 26% の差). 898 フレーム規模のサンプルでは誤差の範囲を超える有意差と判断し, 既定バックエンドを eloftr とする根拠を得た. これにより `notes/PERF_plan.md` の「既定採用の確定は精度比較を待つ」というゲート条件は満たされ, 改善項目2「マッチャーの opt 設定への切替」の前提 (eloftr 既定) も確定した. ただし本比較は SM1 単一動画・単一 seed によるものであり, 他動画での再現性は未確認. 生データ: loftr 姿勢出力 `data/bench_results/ho3d_ours_loftr/SM1/ob_in_cam/`, benchmark 出力 `data/bench_results/ho3d_log_loftr/`, 実行ログ `data/bench_results/loftr_bench_run.log`
- [x] BundleSDF 側 ROS ラッパーノード実装 (`ros/bundlesdf_node/`, rgb+depth+mask 同期購読 → PoseStamped/TF 配信 + `sam3_segmenter` 込みの `bundlesdf.launch`. `docker/docker-compose.yml` + `.devcontainer/devcontainer.json` (compose をラップ) 追加, host networking + ROS_MASTER_URI/ROS_IP 設定. sam3_segmenter とは同一コンテナ (f8a0428 で jammy イメージに統合済み, 別コンテナではない). devcontainer 経由の起動と osx 側 roscore コンテナとの ROS 疎通はユーザー確認済み. `bundlesdf_node`/`sam3_segmenter` のトラッキングパイプライン自体 (rgb/depth/mask 同期→姿勢出力) の実機動作は未検証, 下記項目参照)
- [x] リモート origin 変更 + feat/ros-one-online ブランチの push (2026-07-02, origin=barikata1984/BundleSDF)
- [ ] SAM3 統合後イメージの再ビルドと `Sam3VideoModel` import 検証 (ユーザー指示で中断した分の再開)
- [x] perf CSV (`perf_main.csv`/`perf_nerf.csv`/`perf_nerf_train.csv`) を使ったフルベンチの内訳分析 (Tier 1/3 修正の優先順位決め) (2026-07-03, log 参照)
- [x] 改善項目1: NeRF warm-start (`reuse_weights=True`, `n_step_warm=300`) の実装・効果確認・アブレーション. 壁時計 683.1s→501.4s (-26.6%), nerf_wait 52.7%→35.9%. 軌跡整合ゲートは回転側で不合格 (median 1.885° vs 基準 1.33°) だったが, ユーザーが速度優先でこの回転差 (baseline との相対差, GT に対する絶対精度劣化ではない) を許容し採用を決定. コミット済み (28f25df) (2026-07-03, log 参照)
- [x] VRAM 単調増加 (17→26GB) の要因確認 (キーフレーム蓄積 vs `confs_gpu` リーク). `confs_gpu` の cudaFree 漏れ自体は 2026-07-03 に修正済み (`FeatureManager.cpp:1712`, PERF_plan.md 参照, リビルド・import 確認済み・コミット済み (e894df9)). save_result 非同期化・転送順修正・本修正の 3 件込みの async フルベンチ (`data/out_milk_eloftr_async/`, 1932 フレーム) で VRAM 推移を計測したところ 15.6→16.2→16.8→20.6→27.4GB (ピーク 27.7GB) と baseline (17→26GB) と同型の単調増加が継続し, `confs_gpu` 修正 (RANSAC ペア単位の小さい確保解放) はマクロな VRAM 曲線には効かず, 主因はキーフレーム蓄積 (終盤 262 keyframes) と判明した. ただしこの計測は正式な VRAM ロギング実装 (下記) ではなく, bench 実行時に即興で用意した簡易計測 (`data/bench_results/async_vram.csv`, timestamp/memory_used の 2 列のみ) によるものであり, 完全な確証ではない
- [ ] `bundlesdf_node`/`sam3_segmenter` トラッキングパイプラインの実機検証 (osx 側カメラストリームからの rgb/depth/mask 同期購読 → 実際の姿勢出力まで. devcontainer/compose 経由の ROS 疎通自体は確認済み, パイプライン動作は別)
- [ ] 改善項目1 残り: 同期ポリシーの粒度改善 (`sync_max_delay` の待ち構造変更, 死活バグ修正のみ実施済み). ウォームアップ排除 (ラウンド先頭反復の 2 倍遅延) は原因を 2 プロセス間 GPU 競合と診断済み, MPS 項目 (改善項目3) に委譲, 未実装
- [x] gridencoder の `ModuleNotFoundError` の恒久対処 (2026-07-03). `mycuda/torch_ngp_grid_encoder/grid.py:14` に `sys.path.append(os.path.dirname(code_dir))` を 1 行追加し, `gridencoder.so` の実配置場所である親ディレクトリ `mycuda/` を `sys.path` に加えた (既存の `sys.path.append(code_dir)` は残置). Python ファイルのみの変更で再ビルド不要. `min_rot`/`start_nerf_keyframes` を一時的に下げた 80 フレームのスモークテストで, NeRF ワーカー (`multiprocessing.Process`) が実際に spawn され, `PYTHONPATH` ハックなしで `GridEncoder` のロードに成功し複数ラウンドが完走することを確認した. これまでの `PYTHONPATH=/workspace/mycuda` によるその場しのぎの回避策は不要になった. 未コミット
- [x] VRAM 計測ロジック (`perf_logger.py`/`bench_milk.sh`) がフルベンチ実行時に実際に `perf_main.csv` へ `vram_alloc_mib`/`vram_reserved_mib` 列を出力することの動作確認 (2026-07-03). `BUNDLESDF_PROFILE=1 BUNDLESDF_MATCHER=eloftr scripts/bench_milk.py --max_frames 30` (出力先 `data/out_milk_vramcheck/`, 既存フルベンチ出力とは別ディレクトリ) を実行し, `perf_main.csv` の末尾に `vram_alloc_mib`/`vram_reserved_mib` 列が実際に出力され, 妥当な値 (1 フレーム目: alloc 314.3/reserved 534.0 MiB, 定常: alloc 322.5/reserved 954.0 MiB. alloc はほぼ一定でリーク無し, reserved はキャッシュアロケータのウォームアップ後プラトー) が入ることを確認した. `perf_nerf.csv` もヘッダに両列が正しく含まれることを確認した (ただし 30 フレームではキーフレーム蓄積が 1 個のみで NeRF ラウンド自体は未発火のため, データ行での実値確認はできていない. 列スキーマの正しさは確認済み). `scripts/bench_milk.sh` の `start_gpu_mon`/`stop_gpu_mon` 機構も 12 秒の直接実行で `eloftr_gpu_mem.csv` が正しい形式 (timestamp, memory.used, memory.total) で出力され, 停止後に nvidia-smi のゾンビ残留がないことを確認した. `scripts/perf_stats.py` が新しい CSV を読み込んでもエラーが出ないことも確認した. コード変更なし (動作確認のみ)
- [x] 精度ガードレール構築 (HO3D データ取得 → GT 付きベンチ 1 本完走): `evaluation.zip`/`masks_XMem.zip`/YCB `models` 一式を Google Drive (readme.md 記載の augmented data) から取得し `data/HO3D_v3/` に配置 (`evaluation` は SM1 のみ展開, 他 12 動画は zip に残置し必要時に追加展開可能). `run_ho3d.py --video_dirs .../SM1` (現行既定設定 = eloftr + warm-start `reuse_weights=True`/`n_step_warm=300`) が 898 フレーム全て `ob_in_cam/` を出力してエラーゼロで完走することを確認. 実行過程で upstream 由来のバグ 2 件を発見・修正した: (1) `benchmark_ho3d.py:156` の到達不能コード `args = []` が argparse の `args` を空リストで上書きし `benchmark_one_video` 内の `args.out_dir` 参照で即クラッシュ (該当行を削除), (2) `Utils.py:trimesh_clean()` が `trimesh` 4.x で廃止された `remove_degenerate_faces()`/`remove_duplicate_faces()` を呼んでいた (`update_faces(nondegenerate_faces())`/`update_faces(unique_faces())` に置換). 修正後 `benchmark_ho3d.py` が完走し, SM1 (mustard bottle, 898 フレーム) で ADD 2.20cm / ADD-S 0.98cm / ADD_AUC 78.10% / ADDS_AUC 90.18% / chamfer 0.52cm を取得 (`data/bench_results/ho3d_log/`, gitignore 対象). この数値自体が良好かは upstream 論文値との比較が必要でまだ評価していないが, パイプラインとしては動作確認済み. これにより下記 2 項目 (loftr 比較, warm-start 絶対精度確認) は同じ枠組みを条件を変えて再実行するだけで着手可能になった. 2 件のバグ修正はコミット済み (f40c569).
- [x] save_result 非同期化 (PERF_plan 改善項目2, 精度リスクゼロ) は実装済み (2026-07-03, サブアジェントが `bundlesdf.py` に単一ワーカースレッド + `queue.Queue` で実装. 詳細は PERF_plan.md 参照). ベンチ数値 (`scripts/bench_milk.sh`, async フルベンチ `data/out_milk_eloftr_async/`, warm-start 適用済み baseline `out_milk_eloftr_item1` 比) による効果確認済み: save_result median 21.99ms→0.01ms, フレーム total median 161.7ms→139.3ms (-22.4ms), 壁時計 501.4s→468.3s (-6.6%). 併せて実装済みの転送順修正 (`loftr_wrapper.py`) の効果も確認: loftr_predict median 61.95ms→58.98ms (-3.0ms, 見込み「数ms」と一致)
- [x] VRAM 計測ロジックの実装 (`perf_logger.py` に `_vram_mib()` を追加, `SpanProfiler.flush()` で `vram_alloc_mib`/`vram_reserved_mib` を CSV 末尾に追記. `BUNDLESDF_PROFILE=1` 時のみ有効, CUDA 不可時は nan, 例外を出さない. `perf_main.csv`/`perf_nerf.csv`/`perf_nerf_train.csv` 全てに適用. `scripts/bench_milk.sh` にも `nvidia-smi --query-gpu` をバックグラウンド起動する `start_gpu_mon`/`stop_gpu_mon` を追加, `trap EXIT INT TERM` でゾンビプロセスを回避. smoke test で妥当値を確認済み, 未コミット). **注意**: 上記の async フルベンチはこの実装の完成前後のタイミングで走っており, `data/out_milk_eloftr_async/perf_main.csv` には実際には vram_alloc_mib/vram_reserved_mib 列が入っていない (ヘッダ確認済み). 上記 VRAM 要因確認の数値根拠はこの正式実装ではなく簡易計測による
- [x] 見せかけ高速化チェックスクリプト作成 (`scripts/check_repeated_poses.py`, 未コミット・git 未追跡). async フルベンチの `ob_in_cam` 1932 個で連続フレーム同一姿勢 0 件, 並進 median 0.178cm・回転 median 1.05° を確認し, フレーム間引きによる偽の高速化ではないことを確認した
