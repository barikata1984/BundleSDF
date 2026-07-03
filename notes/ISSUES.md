# ISSUES

- SAM 3 の transformers ストリーミングモードは hotstart heuristics が無効になり, 誤検出・重複トラック増加が公式に警告されている.
- ROS One の apt リポジトリはローリングリリースの単一系列でスナップショット保証がない (対策として Docker イメージ保存を予定しているが未実施).
- SAM3 統合後イメージの再ビルドが未検証: `ros-one.dockerfile` に `transformers==5.12.1`/`huggingface_hub`/`accelerate`/`HF_HOME` を追加し, noble 別コンテナ (`docker/segmenter.dockerfile`) を廃止して jammy 単一コンテナに統合したが, イメージの再ビルドと `Sam3VideoModel` の import 確認はユーザー指示で中断しており未実施 (再開時に要ビルド+import 確認).
- バグレビュー上位10件のうち #1 (pdb) と #9 (dockerfile) 以外は未修正のまま. 特に zmq 経路の memcpy overread と `confs_gpu` のリークが未対応 (一覧は過去の会話ログ/レビュー結果を参照).
