# ISSUES

- OpenCV 4.13.0 + `CUDA_GENERATION=Blackwell` build-arg での実ビルド成功が未検証 (縮小ソースビルドのみで prebuilt 無し).
- SAM 3 の transformers ストリーミングモードは hotstart heuristics が無効になり, 誤検出・重複トラック増加が公式に警告されている.
- ROS One の apt リポジトリはローリングリリースの単一系列でスナップショット保証がない (対策として Docker イメージ保存を予定しているが未実施).
