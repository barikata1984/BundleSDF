# ISSUES

- OpenCV 4.13.0 + `CUDA_GENERATION=Blackwell` build-arg での実ビルド成功が未検証 (縮小ソースビルドのみで prebuilt 無し).
- SAM 3 の transformers ストリーミングモードは hotstart heuristics が無効になり, 誤検出・重複トラック増加が公式に警告されている.
- ROS One の apt リポジトリはローリングリリースの単一系列でスナップショット保証がない (対策として Docker イメージ保存を予定しているが未実施).
- `mycuda/pyproject.toml` の `[build-system] requires` が `torch>=2.6.0` と unpin のため, `pip install -e .` の build isolation が PyPI から CUDA 13.0 版 torch を取得してしまい, コンテナの torch (cu129) と major version が食い違って `_check_cuda_version` が失敗する (`bash build.sh` で毎回再現). 対策候補: `--no-build-isolation` を使うか, pyproject.toml の torch 版を pin する.
- `BundleTrack/src/Utils.h` の `convert3dOrganizedRGB` / `outlierRemovalRadius` / `outlierRemovalStatistic` / `downsamplePointCloud` / `passFilterPointCloud` は仮引数が `typename pcl::PointCloud<PointT>::Ptr` のみで書かれており C++ の非推論コンテキストになっている. `Frame.cpp`/`Bundler.cpp` の呼び出し側が明示的テンプレート実引数を付けていないため GCC 11.4 で `my_cpp` のビルドが失敗する (再現性あり).
- `BundleTrack/src/FeatureManager.cpp` (744/1569/2000 行目) が `pcl::geometry::distance` を呼ぶが `pcl/common/geometry.h` を include していない. PCL 1.12 では `pcl/common/distances.h` からの transitive include が無くなっており `'pcl::geometry' has not been declared` でビルド失敗する.
- `loftr_wrapper.py:48` の `torch.load(ckpt)['state_dict']` が PyTorch 2.6+ のデフォルト `weights_only=True` に引っかかり, `eloftr_outdoor.ckpt` 内の `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` が許可リスト外の global として `UnpicklingError` になる (`weights_only=False` か `torch.serialization.add_safe_globals` が必要).
