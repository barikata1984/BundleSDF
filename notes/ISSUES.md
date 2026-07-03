# ISSUES

- SAM 3 の transformers ストリーミングモードは hotstart heuristics が無効になり, 誤検出・重複トラック増加が公式に警告されている.
- ROS One の apt リポジトリはローリングリリースの単一系列でスナップショット保証がない (対策として Docker イメージ保存を予定しているが未実施).
- SAM3 統合後イメージの再ビルドが未検証: `ros-one.dockerfile` に `transformers==5.12.1`/`huggingface_hub`/`accelerate`/`HF_HOME` を追加し, noble 別コンテナ (`docker/segmenter.dockerfile`) を廃止して jammy 単一コンテナに統合したが, イメージの再ビルドと `Sam3VideoModel` の import 確認はユーザー指示で中断しており未実施 (再開時に要ビルド+import 確認).
- コードレビューの未修正バグ: 上位10件のうち #1 (pdb) と #9 (dockerfile) のみ修正済みで残りは未対応. 全一覧・判定・対応方針は `notes/REVIEW_findings.md` を参照. 特に優先度が高いのは `confs_gpu` リーク (#3, VRAM 単調増加の一因の可能性), zmq memcpy overread (#2), optimizeGPU の FAIL 非 return (#8), percentile 空配列クラッシュ (#5), nerf 子プロセス死活の未確認 (#7).
- 推論速度の未対応改善: Tier 0 (即効 4 件) は対応済みだが Tier 1-3 (C++ の cudaDeviceSynchronize 逐次化, 毎フレーム cudaMalloc, キーフレーム走査の O(N²), MPS 未使用等) は未着手. 全一覧・ベンチ結果・EfficientLoFTR の速度差分析は `notes/PERF_plan.md` を参照.
