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
- [ ] EfficientLoFTR vs 旧 LoFTR の ho3d データセットでの ADD/ADD-S ベンチマーク比較 (劣化なし確認後に既定採用を確定. NeRF warm-start の絶対精度確認も兼ねる予定だったが, warm-start はユーザー判断で速度優先採用済みのため, 本項目にとっては優先度を下げた任意の確認事項). HO3D GT ベンチの枠組み自体は 2026-07-03 に整備済み (`run_ho3d.py`/`benchmark_ho3d.py` が SM1 で完走確認済み, 上記完了項目参照). `BUNDLESDF_MATCHER=loftr` で再実行し既定 (eloftr) の結果 (ADD 2.20cm/ADD-S 0.98cm) と比較すれば着手できる
- [x] BundleSDF 側 ROS ラッパーノード実装 (`ros/bundlesdf_node/`, rgb+depth+mask 同期購読 → PoseStamped/TF 配信 + `sam3_segmenter` 込みの `bundlesdf.launch`. `docker/docker-compose.yml` + `.devcontainer/devcontainer.json` (compose をラップ) 追加, host networking + ROS_MASTER_URI/ROS_IP 設定. sam3_segmenter とは同一コンテナ (f8a0428 で jammy イメージに統合済み, 別コンテナではない). devcontainer 経由の起動と osx 側 roscore コンテナとの ROS 疎通はユーザー確認済み. `bundlesdf_node`/`sam3_segmenter` のトラッキングパイプライン自体 (rgb/depth/mask 同期→姿勢出力) の実機動作は未検証, 下記項目参照)
- [x] リモート origin 変更 + feat/ros-one-online ブランチの push (2026-07-02, origin=barikata1984/BundleSDF)
- [ ] SAM3 統合後イメージの再ビルドと `Sam3VideoModel` import 検証 (ユーザー指示で中断した分の再開)
- [x] perf CSV (`perf_main.csv`/`perf_nerf.csv`/`perf_nerf_train.csv`) を使ったフルベンチの内訳分析 (Tier 1/3 修正の優先順位決め) (2026-07-03, log 参照)
- [x] 改善項目1: NeRF warm-start (`reuse_weights=True`, `n_step_warm=300`) の実装・効果確認・アブレーション. 壁時計 683.1s→501.4s (-26.6%), nerf_wait 52.7%→35.9%. 軌跡整合ゲートは回転側で不合格 (median 1.885° vs 基準 1.33°) だったが, ユーザーが速度優先でこの回転差 (baseline との相対差, GT に対する絶対精度劣化ではない) を許容し採用を決定. 作業ツリーに残置, 未コミット (2026-07-03, log 参照)
- [ ] VRAM 単調増加 (17→26GB) の要因確認 (キーフレーム蓄積 vs `confs_gpu` リーク). `confs_gpu` の cudaFree 漏れ自体は 2026-07-03 に修正済み (`FeatureManager.cpp:1712`, PERF_plan.md 参照, リビルド・import 確認済み・未コミット). 再ベンチによる残存増加分の切り分けは未実施
- [ ] `bundlesdf_node`/`sam3_segmenter` トラッキングパイプラインの実機検証 (osx 側カメラストリームからの rgb/depth/mask 同期購読 → 実際の姿勢出力まで. devcontainer/compose 経由の ROS 疎通自体は確認済み, パイプライン動作は別)
- [ ] 改善項目1 残り: 同期ポリシーの粒度改善 (`sync_max_delay` の待ち構造変更, 死活バグ修正のみ実施済み). ウォームアップ排除 (ラウンド先頭反復の 2 倍遅延) は原因を 2 プロセス間 GPU 競合と診断済み, MPS 項目 (改善項目3) に委譲, 未実装
- [x] 精度ガードレール構築 (HO3D データ取得 → GT 付きベンチ 1 本完走): `evaluation.zip`/`masks_XMem.zip`/YCB `models` 一式を Google Drive (readme.md 記載の augmented data) から取得し `data/HO3D_v3/` に配置 (`evaluation` は SM1 のみ展開, 他 12 動画は zip に残置し必要時に追加展開可能). `run_ho3d.py --video_dirs .../SM1` (現行既定設定 = eloftr + warm-start `reuse_weights=True`/`n_step_warm=300`) が 898 フレーム全て `ob_in_cam/` を出力してエラーゼロで完走することを確認. 実行過程で upstream 由来のバグ 2 件を発見・修正した: (1) `benchmark_ho3d.py:156` の到達不能コード `args = []` が argparse の `args` を空リストで上書きし `benchmark_one_video` 内の `args.out_dir` 参照で即クラッシュ (該当行を削除), (2) `Utils.py:trimesh_clean()` が `trimesh` 4.x で廃止された `remove_degenerate_faces()`/`remove_duplicate_faces()` を呼んでいた (`update_faces(nondegenerate_faces())`/`update_faces(unique_faces())` に置換). 修正後 `benchmark_ho3d.py` が完走し, SM1 (mustard bottle, 898 フレーム) で ADD 2.20cm / ADD-S 0.98cm / ADD_AUC 78.10% / ADDS_AUC 90.18% / chamfer 0.52cm を取得 (`data/bench_results/ho3d_log/`, gitignore 対象). この数値自体が良好かは upstream 論文値との比較が必要でまだ評価していないが, パイプラインとしては動作確認済み. これにより下記 2 項目 (loftr 比較, warm-start 絶対精度確認) は同じ枠組みを条件を変えて再実行するだけで着手可能になった. 2 件のバグ修正は作業ツリーに残置, 未コミット.
- [ ] save_result 非同期化 (PERF_plan 改善項目2, 精度リスクゼロ) は実装済み (2026-07-03, サブアジェントが `bundlesdf.py` に単一ワーカースレッド + `queue.Queue` で実装. 詳細は PERF_plan.md 参照). ベンチ数値 (`scripts/bench_milk.sh`) による効果確認は未実施
