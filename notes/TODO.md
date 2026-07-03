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
- [ ] EfficientLoFTR vs 旧 LoFTR の ho3d データセットでの ADD/ADD-S ベンチマーク比較 (劣化なし確認後に既定採用を確定)
- [ ] BundleSDF 側 ROS ラッパーノード実装 (rgb+depth+mask を同期購読し PoseStamped/TF を配信)
- [x] リモート origin 変更 + feat/ros-one-online ブランチの push (2026-07-02, origin=barikata1984/BundleSDF)
- [ ] SAM3 統合後イメージの再ビルドと `Sam3VideoModel` import 検証 (ユーザー指示で中断した分の再開)
- [ ] perf CSV (`perf_main.csv`/`perf_nerf.csv`/`perf_nerf_train.csv`) を使ったフルベンチの内訳分析 (Tier 1/3 修正の優先順位決め)
- [ ] VRAM 単調増加 (17→26GB) の要因確認 (キーフレーム蓄積 vs `confs_gpu` リーク)
