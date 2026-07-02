# 議事録: BundleSDF の ROS One 対応と依存スタック刷新 (2026-07-02)

参加: ユーザー + Claude Code (調査サブエージェント: Sonnet ×5, 実装チーム: interface-hub / f1-container / f2-segmenter / f3-matcher)

## 1. 背景と目的

- BundleSDF (NVlabs fork) を ROS-O / ROS One 向けパッケージとしてビルド可能にし, オンラインで 6-DoF 姿勢をトピック配信できるようにする.
- ホスト環境: Ubuntu 24.04 (noble), RTX 5090 (Blackwell, sm_120), ROS/CUDA/torch は未導入.
- 卒業後にラボの後任が維持できること (再現可能なビルドレシピ) を重視する.

## 2. 環境調査で確定した制約

- RTX 5090 (sm_120) は CUDA 12.8 以上 + PyTorch 2.7.0 以上 (cu128 wheel) が必須. リポジトリ現行スタック (CUDA 12.6 + torch 2.6) では動作しない.
- ROS One (ros.packages.techfak.net) は jammy / noble 両 suite を提供 (jammy 1009 パッケージ, ros-one-ros-base / cv-bridge / catkin 等を確認済み).
- ros-one-cv-bridge は system OpenCV (jammy 4.5.4) にリンクするため, 自前 CUDA OpenCV との同一プロセス混在は不可 → cv_bridge は使わない方針.

## 3. 依存ライブラリの調査結果 (prebuilt 化)

ソースビルドを残すのは OpenCV / BundleTrack 本体 / mycuda の 3 つのみに削減できる.

| 依存 | 結論 |
|---|---|
| CMake / Eigen / pybind11 / yaml-cpp | jammy apt で完全代替 (3.22.1 / 3.4.0 / 2.9.1 / 0.7.0) |
| PCL | jammy apt 1.12.1 で代替. ただし PCL 1.11 の std::shared_ptr 移行に伴い Utils.h/cpp のシグネチャ修正が必要 (機械的) |
| torch | 2.8.0+cu129 公式 wheel (cp310) |
| kaolin | 0.18.0 公式 prebuilt wheel (torch-2.8.0_cu129 インデックスに cp310 実在確認). 上限制約はここ (torch ≤2.8.0 / ≤cu129). CUDA 13 系は kaolin が追随するまで不可 |
| pytorch3d | 実使用は se3_exp_map ほか transforms 3 関数のみ (pure PyTorch) → ベンダリングで依存削除 |
| OpenCV w/ CUDA | prebuilt は全滅 (conda-forge / NVIDIA / PPA / cudawarped いずれも C++ リンク用途不可) → 縮小ソースビルド |

## 4. 主要な決定事項

1. **実行環境**: Ubuntu 22.04 (jammy) コンテナ + ROS One jammy. ホスト (noble) の roscore とは --net=host で接続.
2. **成立する組み合わせ** (古→新): CUDA 12.8+torch 2.7.0 / 2.7.1 / 2.8.0, CUDA 12.9+torch 2.8.0. **採用は #4 (CUDA 12.9 + torch 2.8.0+cu129 + kaolin 0.18.0)**.
3. **OpenCV は 4.13.0 を採用** (手持ち 4.12 成果物の再利用ではなく). 理由: Blackwell 公式対応 + CUDA 13.0 サポート + 4.x 最終盤. 引き継ぎのためバイナリ再利用ではなく再現可能なレシピ (dockerfile) + レジストリキャッシュ運用とする. アーキ指定は `CUDA_GENERATION=Blackwell` build-arg (未指定は全アーキビルドになるので不可, `Auto` は docker build 中に GPU 不可視でフォールバックするため不採用).
4. **BundleSDF 本体は catkin 化しない**. 薄い rospy ラッパー方式 (ラッパーノード自体は今回スコープ外).
5. **セグメンタは SAM 3** (欠落していた XMem = GPL の代替). SAM 3 は Python ≥3.12 要求のため **noble + ROS One noble の別コンテナ**. transformers のストリーミング API を使用. concept API はテキストプロンプト専用のため **bbox 初期化は除外** (必要になったら Sam3TrackerVideo (PVS 系) へ差し替え).
6. **特徴点マッチは EfficientLoFTR に置き換え** (既定). 旧 LoFTR はベンチ比較用に残し env `BUNDLESDF_MATCHER` で切替.
7. `opencv2/rgbd.hpp` include は実使用ゼロと判明 → 削除し contrib 要件から rgbd を除外.

## 5. 実装結果 (feat/ros-one-online, 未コミット)

インターフェース契約: `notes/CONTRACT_ros_one_online.md` (1 ラウンドで凍結).

- **F1 コンテナ + ビルド修正**: `docker/ros-one.dockerfile` (CUDA 12.9.2-devel-ubuntu22.04, OpenCV 4.13.0 縮小 CUDA ビルド → /opt/opencv-cuda, ROS One jammy, torch/kaolin/loguru). PCL Ptr 化 (Utils.h/cpp, Frame.cpp の pcl::make_shared 化), rgbd include 削除, CMakeLists に CUDA≥12.8 → sm_120 分岐, build.sh の python3.10 直書き除去, `pytorch3d_transforms/` ベンダリング + nerf_helpers.py の import 差し替え.
- **F2 SAM 3 セグメンタ**: `ros/sam3_segmenter/` catkin パッケージ (`~image_in` → `/sam3/mask` mono8 0/255, header 完全継承, `~text_prompt` 初期化, cv_bridge 不使用) + `docker/segmenter.dockerfile` (CUDA 12.8.1-runtime-ubuntu24.04 + torch 2.9.1/cu128 + transformers 5.12.1, HF_HOME volume).
- **F3 EfficientLoFTR**: `BundleTrack/EfficientLoFTR/` ベンダリング (upstream ffd4a46, Apache-2.0, detect_NaN スタブ), `loftr_wrapper.py` 切替実装 (reparameter() 必須実行, 400→416 下端右端パディング, predict 入出力契約維持, bundlesdf.py 無改変).

検証済み: 全 Python の py_compile, C++ 変更の grep 一貫性 (boost::shared_ptr / make_shared / rgbd 残存ゼロ), bundlesdf.py・旧 dockerfile・BundleTrack/LoFTR の無改変, ベース image タグと wheel の実在.

## 6. 未実施 (次アクション)

- [ ] コンテナビルド (`docker build -f docker/ros-one.dockerfile .` / `segmenter.dockerfile`) と GHCR 等への push (キャッシュ運用)
- [ ] 重み配置: eloftr_outdoor.ckpt (Google Drive), SAM 3 (HF gated, 利用承認 + hf auth login), LoFTR outdoor_ds.ckpt
- [ ] コンテナ内での実行時検証 (import, 実マッチング, SAM 3 ストリーミング)
- [ ] EfficientLoFTR vs LoFTR の ho3d ベンチ比較 (ADD/ADD-S で劣化なしを確認してから既定採用を確定)
- [ ] BundleSDF 側 ROS ラッパーノード (rgb+depth+mask 同期購読 → PoseStamped/TF 配信) の実装
- [ ] リモート origin を git@github.com:barikata1984/BundleSDF.git に変更してブランチを push (作業途中で中断)

## 7. 留保事項

- SAM 3 の transformers ストリーミングモードは hotstart heuristics が無効 (誤検出・重複トラック増を公式が警告).
- OpenCV 4.13.0 + CUDA_GENERATION=Blackwell の実ビルド成功はコンテナビルドで初検証となる.
- ROS One apt はローリング単一系列でスナップショット保証なし → ビルド済みイメージの保存がリスクヘッジ.
