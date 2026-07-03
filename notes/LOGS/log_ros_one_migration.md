# 作業ログ: BundleSDF の ROS One 移行

## 2026-07-02

目的: BundleSDF を ROS One (ROS-O) 向けにビルド可能にし, オンライン 6-DoF 姿勢配信への道を作る.
詳細は `notes/MINUTES_2026-07-02_ros_one_online.md` 参照.

### 調査確定事項

- RTX 5090 (sm_120) は CUDA 12.8+ / torch 2.7+ 必須.
- ROS One は jammy / noble 両対応. C++ 依存 (cmake/eigen/pybind11/yaml-cpp/pcl) は jammy apt で代替可
  (PCL 1.12 は std::shared_ptr 移行対応が必要).
- kaolin 0.18.0 が torch ≤2.8.0 / cu129 の上限を決める.
- pytorch3d は transforms 3 関数のみ使用 (pure PyTorch でベンダリング可能).
- OpenCV w/CUDA の prebuilt は存在せず縮小ソースビルドのみ.
- `opencv2/rgbd.hpp` include は未使用と判明.

### 決定事項

- jammy コンテナ + ROS One jammy.
- CUDA 12.9 + torch 2.8.0+cu129 + kaolin 0.18.0 公式 wheel.
- OpenCV 4.13.0 縮小ビルド (`CUDA_GENERATION=Blackwell` build-arg).
- pytorch3d はベンダリング.
- セグメンタは SAM 3 (別コンテナ, noble + Python 3.12, transformers streaming, text prompt のみ,
  bbox は将来 `Sam3TrackerVideo` で対応).
- 特徴点マッチは EfficientLoFTR を既定とし, env `BUNDLESDF_MATCHER` で旧 LoFTR に切替可能とする.

### 実装結果

feat/ros-one-online ブランチ (未コミット). インターフェース契約は `notes/CONTRACT_ros_one_online.md`.

- `docker/ros-one.dockerfile`, `docker/segmenter.dockerfile` を作成.
- `ros/sam3_segmenter/` catkin パッケージを作成.
- `BundleTrack/EfficientLoFTR/` ベンダリング + `loftr_wrapper.py` にバックエンド切替を実装.
- `pytorch3d_transforms/` ベンダリング + `nerf_helpers.py` の import 差し替え.
- PCL 1.12 対応 (`Utils.h/cpp`, `Frame.cpp`), rgbd include 削除, CMakeLists の sm_120 分岐,
  `build.sh` の python3.10 直書き除去.
- 検証範囲: `py_compile` と grep による一貫性確認まで. コンテナビルド・実行時検証は未実施.

## 2026-07-03

目的: ミルクカートンデモ動画でのオンライン姿勢推定ベンチを"すぐ実行できる状態"まで準備する
(実ベンチ実行はスコープ外). `bundlesdf:ros-one` イメージから永続コンテナ `bundlesdf_bench` を起動し,
`bash build.sh` を実行して検証した.

### データ・重み配置

- ミルクデモ動画 (Google Drive `1akutk_Vay5zJRMr3hVzZ7s69GT4gxuWN`, gdown 経由, 1.28GB) を
  `data/2022-11-18-15-10-24_milk/` に展開. rgb/depth/masks 各 1932 枚, cam_K.txt 確認. `data/` は
  既に `.gitignore` 済み (追記不要).
- `eloftr_outdoor.ckpt` (193MB) を `BundleTrack/EfficientLoFTR/weights/` に, `outdoor_ds.ckpt`
  (46MB) を `BundleTrack/LoFTR/weights/` に配置. 両方 git 管理外であることを確認済み.
- ホストの system python3 に pip が無かったため (`ensurepip` は Debian/Ubuntu で無効化), venv
  経由で `gdown` を導入して回避した.

### コンテナ内ビルド失敗 (3件, 再現性あり: build.sh を2回実行し同一の失敗を確認)

1. **mycuda (`common` 拡張) の pip ビルドが CUDA バージョン不一致で失敗**:
   `mycuda/pyproject.toml` の `[build-system] requires` に `torch>=2.6.0` と unpin で書かれている
   ため, `pip install -e .` の build isolation が **PyPI から別の torch (CUDA 13.0 でコンパイル
   されたもの) を取得**してしまい, コンテナに実際に入っている torch (2.8.0+cu129, nvcc 12.9) と
   食い違う. `torch.utils.cpp_extension._check_cuda_version` が major version 不一致 (12 vs 13) で
   `RuntimeError` を送出する. コンテナの site-packages 側の torch 自体は cuda 12.9 で一貫している
   ことを個別に確認済みなので, 原因は isolation 環境側の torch 取得にある.
2. **BundleTrack (my_cpp) の C++ コンパイル失敗, テンプレート実引数推論エラー**:
   `Utils.h` の `convert3dOrganizedRGB` / `outlierRemovalRadius` / `outlierRemovalStatistic` /
   `downsamplePointCloud` / `passFilterPointCloud` はいずれも仮引数型が
   `typename pcl::PointCloud<PointT>::Ptr` のみで `PointT` を含む形になっており, これは C++ の
   非推論コンテキストに該当する. `Frame.cpp`/`Bundler.cpp` の呼び出し側は明示的テンプレート実引数を
   付けていないため, GCC 11.4 が `PointT` を推論できず "no matching function" / "template-id ...
   does not match any template declaration" で失敗する.
3. **BundleTrack (my_cpp) の C++ コンパイル失敗, `pcl::geometry` 未宣言**:
   `FeatureManager.cpp` の 3 箇所 (744, 1569, 2000 行目) が `pcl::geometry::distance` を呼ぶが,
   `pcl/common/geometry.h` を直接 include していない. PCL 1.12 では `pcl/common/distances.h`
   (Utils.h が include 済み) からの transitive include が無くなっており, 未宣言エラーになる.

いずれもソース修正はスコープ外のため実施していない (`notes/ISSUES.md` に追記).

### GPU / ランタイム smoke 結果

- `torch.cuda` : True, `NVIDIA GeForce RTX 5090`, 行列積 smoke OK.
- `kaolin.__version__` : `0.18.0`, import OK.
- `my_cpp` import: 失敗 (上記ビルド失敗のため `BundleTrack/build/my_cpp*.so` が生成されていない).
- `LoftrRunner(backend=eloftr)` load: 失敗. `loftr_wrapper.py:48` の
  `torch.load(ckpt)['state_dict']` が PyTorch 2.6+ のデフォルト `weights_only=True` に引っかかり,
  `eloftr_outdoor.ckpt` に含まれる `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
  という許可リスト外の global が原因で `UnpicklingError` になる (4件目の新規発見の問題).

### ベンチスクリプト

`run_custom.py` は変更せず, `scripts/bench_milk.py` (フレーム毎 track() 時間計測),
`scripts/compare_poses.py` (eloftr/loftr 間の姿勢軌跡相対比較), `scripts/bench_milk.sh` (両
バックエンドを回してその 2 つを呼ぶオーケストレーション) を新規作成. 出力先は
`data/bench_results/` (gitignore 対象). 構文チェックのみ実施 (`bash -n` / `py_compile`), 実行は
上記ビルド失敗により現状不可.

### ビルドエラー修正とミルクベンチ完走

上記 3 件のソース修正 (mycuda pyproject.toml の torch pin, Utils.h テンプレート実引数の明示, `FeatureManager.cpp` への `pcl/common/geometry.h` include 追加) と, 新規発見の 4 件目 (`loftr_wrapper.py` の `torch.load` を `weights_only=False` に対応) を実施し, `bash build.sh` のビルド成功を確認した. これを受けて `scripts/bench_milk.sh` を実行し, `data/2022-11-18-15-10-24_milk/` の全 1932 フレームを eloftr/loftr 双方でエラーゼロで完走させた.

速度: eloftr の track() median は 162.1ms, loftr は 177.7ms で eloftr が約 9% 速い. mean はどちらも NeRF/BA の裾 (105 フレームが 1–5 秒かかる) に支配され, マッチャー間の差はほぼ現れない. 序盤 100 フレームは 18.7fps 程度だが, キーフレーム蓄積に伴い単調減速する. VRAM は 17GB から 26GB まで単調増加する (要因未特定, 下記課題参照).

姿勢整合性: eloftr と loftr の軌跡比較で並進誤差 median 0.16cm, 回転誤差 median 1.33° (max 19.6°). 結果一式は `data/bench_results/` に出力.

速度差が小さい理由を分析した: マッチャーはフレーム時間全体の中では少数派の処理でありアムダールの法則により全体速度への寄与が頭打ちになる, ベンチの入力解像度が 400px 級の小入力レジームでマッチャー自体のコストが元々小さい, ベンチ設定が full+mp (multiprocessing) であること, 裾を作っている NeRF/BA の遅さはマッチャーと無関係であること, の 4 点が要因である.

Tier 0 の性能修正・ビルドブロッカー修正・ベンチスクリプト一式を 6 コミットにまとめ, コミット・push 済み.

### SAM3 が Python 3.10 で動作可能と判明, コンテナ統合

SAM3 は Python 3.10 で動作することが判明した. 実ゲートは `transformers==5.12.1` の `requires_python>=3.10` であり, sam3 の README に書かれている"Python 3.12+ 必須"という記載は pyproject.toml の実際の要求と矛盾しており誤りと判断した.

この判明を受けて, noble ベースの別コンテナ (`docker/segmenter.dockerfile`) を廃止し, jammy 単一コンテナへ統合した. `ros-one.dockerfile` に `transformers==5.12.1`, `huggingface_hub`, `accelerate`, `HF_HOME` を追加し, `docker/segmenter.dockerfile` を削除, `ros/sam3_segmenter/` の README を更新した. 統合後イメージの再ビルド検証はユーザー指示により中断し, 未実施 (`Sam3VideoModel` の import 確認を含め再開時に要対応, `notes/ISSUES.md` に追記済み).

### プロセス構造の分解調査

Opus によるサブエージェント調査で, メインプロセスおよび NeRF ワーカーの全サブステップを file:line 付きで確定した. 副産物として, C++ 側の `Bundler::runNerf` と zmq の 3 ポート構成は dead code であること, C++ 側の `Utils::Timer` はどこにもインスタンス化されておらず `TIMER=1` を立てても出力が無いことが判明した.

### per-stage 時間計測の実装

`perf_logger.py` を新規作成し, `bundlesdf.py` に 32 箇所の計測スパンを追加, `nerf_runner.py` に `train` の内訳計測を追加, `bench_milk.sh` に `BUNDLESDF_PROFILE=1` を追加した. 出力は各 `out_folder` 配下の `perf_main.csv` / `perf_nerf.csv` / `perf_nerf_train.csv`. 30 フレームでの実測では residual (計測されない残余時間) が 2–4%, 計測を無効にした場合のオーバーヘッドはゼロであることを確認した. 定常フレームでの支配項は `loftr_predict` であることが分かった.

### readme.md 更新

`readme.md` 冒頭に本 fork の概要セクションを追加した (スタック構成, upstream からの変更点, ビルド/ベンチ手順).
