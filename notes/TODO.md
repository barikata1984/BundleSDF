# TODO

## 完了

- [x] コンテナ定義ファイル作成 (`docker/ros-one.dockerfile`, `docker/segmenter.dockerfile`)
- [x] 依存ベンダリング (`pytorch3d_transforms/`, `BundleTrack/EfficientLoFTR/` + `loftr_wrapper.py` 切替)
- [x] PCL 1.12 対応 (`Utils.h/cpp`, `Frame.cpp` の shared_ptr 化) / rgbd include 削除 / CMakeLists sm_120 分岐 / build.sh 修正

## 未完了

- [ ] コンテナ2種 (`ros-one.dockerfile`, `segmenter.dockerfile`) のビルドと GHCR push
- [ ] 重み配置 (eloftr_outdoor.ckpt=Google Drive, SAM3=HF gated リポジトリ要承認申請, LoFTR outdoor_ds.ckpt)
- [ ] コンテナ内実行時検証 (import 確認, 実マッチング動作確認, SAM3 streaming 動作確認)
- [ ] EfficientLoFTR vs 旧 LoFTR の ho3d データセットでの ADD/ADD-S ベンチマーク比較 (劣化なし確認後に既定採用を確定)
- [ ] BundleSDF 側 ROS ラッパーノード実装 (rgb+depth+mask を同期購読し PoseStamped/TF を配信)
- [x] リモート origin 変更 + feat/ros-one-online ブランチの push (2026-07-02, origin=barikata1984/BundleSDF)
