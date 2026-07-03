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
- [ ] EfficientLoFTR vs 旧 LoFTR の ho3d データセットでの ADD/ADD-S ベンチマーク比較 (劣化なし確認後に既定採用を確定. NeRF warm-start の絶対精度確認も兼ねる予定だったが, warm-start はユーザー判断で速度優先採用済みのため, 本項目にとっては優先度を下げた任意の確認事項)
- [x] BundleSDF 側 ROS ラッパーノード実装 (`ros/bundlesdf_node/`, rgb+depth+mask 同期購読 → PoseStamped/TF 配信 + `sam3_segmenter` 込みの `bundlesdf.launch`. `docker/docker-compose.yml` + `.devcontainer/devcontainer.json` (compose をラップ) 追加, host networking + ROS_MASTER_URI/ROS_IP 設定. sam3_segmenter とは同一コンテナ (f8a0428 で jammy イメージに統合済み, 別コンテナではない). devcontainer 経由の起動と osx 側 roscore コンテナとの ROS 疎通はユーザー確認済み. `bundlesdf_node`/`sam3_segmenter` のトラッキングパイプライン自体 (rgb/depth/mask 同期→姿勢出力) の実機動作は未検証, 下記項目参照)
- [x] リモート origin 変更 + feat/ros-one-online ブランチの push (2026-07-02, origin=barikata1984/BundleSDF)
- [ ] SAM3 統合後イメージの再ビルドと `Sam3VideoModel` import 検証 (ユーザー指示で中断した分の再開)
- [x] perf CSV (`perf_main.csv`/`perf_nerf.csv`/`perf_nerf_train.csv`) を使ったフルベンチの内訳分析 (Tier 1/3 修正の優先順位決め) (2026-07-03, log 参照)
- [x] 改善項目1: NeRF warm-start (`reuse_weights=True`, `n_step_warm=300`) の実装・効果確認・アブレーション. 壁時計 683.1s→501.4s (-26.6%), nerf_wait 52.7%→35.9%. 軌跡整合ゲートは回転側で不合格 (median 1.885° vs 基準 1.33°) だったが, ユーザーが速度優先でこの回転差 (baseline との相対差, GT に対する絶対精度劣化ではない) を許容し採用を決定. 作業ツリーに残置, 未コミット (2026-07-03, log 参照)
- [ ] VRAM 単調増加 (17→26GB) の要因確認 (キーフレーム蓄積 vs `confs_gpu` リーク)
- [ ] `bundlesdf_node`/`sam3_segmenter` トラッキングパイプラインの実機検証 (osx 側カメラストリームからの rgb/depth/mask 同期購読 → 実際の姿勢出力まで. devcontainer/compose 経由の ROS 疎通自体は確認済み, パイプライン動作は別)
- [ ] 改善項目1 残り: 同期ポリシーの粒度改善 (`sync_max_delay` の待ち構造変更, 死活バグ修正のみ実施済み). ウォームアップ排除 (ラウンド先頭反復の 2 倍遅延) は原因を 2 プロセス間 GPU 競合と診断済み, MPS 項目 (改善項目3) に委譲, 未実装
- [ ] 次セッション冒頭: 精度ガードレール構築をバックグラウンドで開始する (HO3D データ取得 → GT 付きベンチ 1 本完走. `data_reader.py` の `HO3D_ROOT` は変更済み). 並行して save_result 非同期化 (PERF_plan 改善項目2, 精度リスクゼロ) を実施する (2026-07-03 方針決定: 精度に触る変更に着手する前に測る道具を先に立てる)
