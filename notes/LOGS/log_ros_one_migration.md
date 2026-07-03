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

### stale な前提の訂正: SAM3 セグメンタは別コンテナではない

`notes/CONTRACT_ros_one_online.md` と `notes/MINUTES_2026-07-02_ros_one_online.md` は, SAM 3 セグメンタが noble + Python 3.12 の別コンテナ (`docker/segmenter.dockerfile`) で動く前提のまま残っている. しかし前節 (SAM3 が Python 3.10 で動作可能と判明, コンテナ統合) で述べた通り, `docker/segmenter.dockerfile` は既に廃止され, SAM 3 の依存 (`transformers`, `huggingface_hub`, `accelerate`, `HF_HOME`) は `docker/ros-one.dockerfile` に統合済みである (commit f8a0428). したがって `sam3_segmenter` と本節以降で実装するトラッカーノードは同一コンテナで動く. エージェントは ROS ラッパーノード実装の作業に着手した際, 当初この点を見落として別コンテナ前提で案内したが, ユーザーの指摘を受けて `git log`/`git show f8a0428` で確認し訂正した. CONTRACT/MINUTES 側の記述更新は本セッションのスコープ外としたため, 該当箇所については本ログの記載を正とする.

### ROS ラッパーノード実装 (`ros/bundlesdf_node/`)

`notes/TODO.md` の未着手項目 (BundleSDF 側 ROS ラッパーノード実装) に対応する新規 catkin パッケージ `ros/bundlesdf_node/` を作成した.

- `scripts/bundlesdf_node.py`: `~rgb_in`/`~depth_in`/`~camera_info_in`/`~mask_in` を `message_filters.ApproximateTimeSynchronizer` (slop=0.05s) で同期購読する. デコードは cv_bridge を使わず `np.frombuffer` で行い (本 fork の既存方式に合わせた), 同期フレームごとに既存の `bundlesdf.py` を無改造のまま `BundleSdf.run()` で呼ぶ. 姿勢は `BundleTrack` (C++) の出力である `<out_folder>/ob_in_cam/<id_str>.txt` から読み戻す (`scripts/compare_poses.py` が既に読んでいるのと同じパス). 読み戻した姿勢は `geometry_msgs/PoseStamped` として `~object_pose` に配信し, 同時に `tf2_ros.TransformBroadcaster` で TF (既定の子フレーム名 `tracked_object`) も配信する. 回転行列からクォータニオンへの変換には ROS 標準の `tf.transformations.quaternion_from_matrix` ではなく `scipy.spatial.transform.Rotation` を使った. `tf` パッケージのクォータニオン成分順序が ROS の規約と異なる上, `ros-one.dockerfile` には `tf` 自体がインストールされていない一方, `tf2_ros` はインストール済みだからである.
- `package.xml`/`CMakeLists.txt` は `ros/sam3_segmenter` の構成に合わせた.
- `launch/bundlesdf_node.launch`: トラッカー単体の起動用. デフォルトのトピック名は osx 側の実際のトピック名 (`/d455_1/color/image_rect`, `/d455_1/aligned_depth_to_color/image_raw`, `/d455_1/color/camera_info_rect`) と, `notes/CONTRACT_ros_one_online.md` §2 で凍結済みの `/sam3/mask` 契約に合わせた.
- `launch/bundlesdf.launch`: `use_segmenter` 引数 (既定 true) で `sam3_segmenter.launch` を `<include>` する統合起動ファイル. `target_object` を SAM 3 のテキストプロンプトとして転送する. sam3_segmenter とトラッカーノードは同一コンテナで動くことが確定しているため, この 1 ファイルで両方を起動できる.
- `README.md`: トピック契約表, パラメータ, 起動コマンド (単体/統合) を記載し, スコープ外の項目 (メッシュ/点群配信, ロボット状態の購読=`with_robot` トグル, トラッキング消失後の自動再初期化 — 消失フレームは単に FAIL のまま残る) を明記した.
- 検証は `python3 -m py_compile` と XML パースのみで, 実行時動作 (rgb/depth/mask が実際に同期して届き, 姿勢が出るか) は本セッションのエージェント作業では確認していない.

### `docker/docker-compose.yml` 追加

単一サービス `bundlesdf` を定義した. `network_mode: host` + `ipc: host`. GPU 割り当ては非推奨の `runtime: nvidia` ではなく `deploy.resources.reservations.devices` (driver nvidia, capabilities [gpu]) で行う. 環境変数に `ROS_MASTER_URI=http://127.0.0.1:11311` と `ROS_IP=127.0.0.1` を設定した. host networking 下で共有ループバック越しにコンテナをまたいだ ROS ノード発見を成立させるための設定で, `ROS_IP` はイメージに焼き込まれた `ROS_HOSTNAME` を上書きする. 加えて `NVIDIA_DISABLE_REQUIRE=1`, リポジトリを `/workspace` にマウントするほか `/home`/`/tmp`/`/mnt` もマウントし, `/hf_cache` は名前付きボリュームにした. ビルドは既存の `docker/ros-one.dockerfile` (`context: ..`) から行う. `docker compose config` で検証した.

### ROS1 シングルマスター前提の指摘

osx 側 (roscore と RealSense d455 カメラを動かす別コンテナ, 別マシン/セッション) オペレータの計画は, 各コンテナが個別に `roscore` を立てるというものだった. これを問題として指摘した. ROS1 はシングルマスターであり, 別々の master に登録されたノードは互いのトピックを発見できない (劣化ではなく完全に不可視になる). 別 master 同士を橋渡しするには `multimaster_fkie` (`master_discovery`/`master_sync`) が必要だが, 今回の構成には含まれていない. ユーザーはこの指摘を確認し, osx 側を単一の共有 `roscore` に揃える方針で合意した.

### `.devcontainer/devcontainer.json` 追加

既存の `docker/docker-compose.yml` をラップする形で追加した (設定を二重に持たせない). `dockerComposeFile: ["../docker/docker-compose.yml"]`, `service: bundlesdf`, `workspaceFolder: /workspace`. `ros-one.dockerfile` は非 root ユーザーを持たず root で動く前提のため `remoteUser` は指定していない. `shutdownAction: "none"` とし, VS Code ウィンドウを閉じても roscore に依存する他セッション/exec シェルのプロセスを巻き込んで落とさないようにした. 拡張機能は最小限 (Python/Pylance/ruff, C++ tools + CMake Tools [`BundleTrack` の C++/CUDA ビルド用], `ms-iot.vscode-ros`, Docker, GitLens, YAML) に絞った. ユーザーの他リポジトリ (`docker-devcontainer-template`, `nerfstudio` 等) が使う pixi + 非 root ユーザーの devcontainer テンプレートは意図的に踏襲していない. 本リポジトリの `ros-one.dockerfile` は root ユーザー・非 pixi の構成であり, テンプレートをそのまま持ち込むと構成が食い違うと判断したためである.

### devcontainer 経由での動作確認 (ユーザー確認)

ユーザーが devcontainer を開き, compose サービスに対して起動できること, および osx 側コンテナとの ROS 通信が確立することを確認した ("ナイス. 行けたわ."). 確認できたのは devcontainer + compose + host networking 下でのクロスコンテナ ROS 疎通までであり, `bundlesdf_node`/`sam3_segmenter` のトラッキングパイプライン自体 (実際の rgb/depth/mask 同期購読から姿勢出力まで) を実機で動かした確認ではない. この切り分けは `notes/TODO.md` に反映した.

### プロセス構造の分解調査

Opus によるサブエージェント調査で, メインプロセスおよび NeRF ワーカーの全サブステップを file:line 付きで確定した. 副産物として, C++ 側の `Bundler::runNerf` と zmq の 3 ポート構成は dead code であること, C++ 側の `Utils::Timer` はどこにもインスタンス化されておらず `TIMER=1` を立てても出力が無いことが判明した.

### per-stage 時間計測の実装

`perf_logger.py` を新規作成し, `bundlesdf.py` に 32 箇所の計測スパンを追加, `nerf_runner.py` に `train` の内訳計測を追加, `bench_milk.sh` に `BUNDLESDF_PROFILE=1` を追加した. 出力は各 `out_folder` 配下の `perf_main.csv` / `perf_nerf.csv` / `perf_nerf_train.csv`. 30 フレームでの実測では residual (計測されない残余時間) が 2–4%, 計測を無効にした場合のオーバーヘッドはゼロであることを確認した. 定常フレームでの支配項は `loftr_predict` であることが分かった.

### readme.md 更新

`readme.md` 冒頭に本 fork の概要セクションを追加した (スタック構成, upstream からの変更点, ビルド/ベンチ手順).

### perf CSV フルベンチ内訳分析 (eloftr, 1932 フレーム)

`data/out_milk_eloftr/perf_main.csv` (1932 行) と `perf_nerf.csv` (111 ラウンド), `perf_nerf_train.csv` を分析した.

- 全体 683.1s (2.83fps). total_ms は median 163.5ms, p90 228ms, p95 2545ms, max 4967ms.
- 裾の帰属: p90 超の 194 フレームが壁時計の 60.0% を占め, その 87.8% が `nerf_wait` (NeRF 同期待ち). `nerf_wait` は 112 フレームに集中し合計 359.9s = 全体の 52.7%. 以前"NeRF/BA の裾"と記録したが, BA (`optimize_gpu`) は median 21.1ms / max 49.1ms で裾に寄与しておらず, 裾は NeRF 待ちのみである.
- NeRF ワーカーは 111 ラウンド合計 696.8s で壁時計とほぼ等しく, セッション全体で飽和稼働している (トラッキング側が sync_max_delay=3 のバックプレッシャで律速). ラウンド内訳: train 85.9% (render 44.8% + backward 42.2%), `runner_build` 13.0% (median 793ms, 毎ラウンド再構築). 各ラウンド先頭の 50-iter bin は 812ms で以降の 439ms の約 2 倍 (ウォームアップコスト).
- 定常フレーム (p90 以下, 1738 フレーム): `find_corres_ref`+`find_corres_local` ≈ 85ms/フレーム (内訳: `loftr_predict` 61.4ms, `ransac` 14.8ms, `get_pairs` 6.3ms), `save_result` 21.6ms, `optimize_gpu` 21.1ms, `select_kf` 15.5ms.
- 経時劣化: total median は 5 分割で 138→180ms に増加. 増分は `select_kf` (2.6→24.9ms) と `save_result` (15.8→28.9ms) でほぼ説明でき, `find_corres`/`loftr_predict`/`optimize_gpu` は横ばい. キーフレーム蓄積に対する線形スケーリングが原因で, VRAM 単調増加 (17→26GB) と同根の可能性が高い.
- 修正の優先順位提案: (1) NeRF ラウンドスループット改善 (`runner_build` の再利用 + ラウンド頭のウォームアップ排除で約 -18%/ラウンド, さらに train 反復数削減や warm-start は精度検証込みで), (2) `save_result` の非同期化 (定常 median -22ms, 実装容易), (3) `select_kf` のスケーリング対策 (終盤 -25ms, VRAM 問題と併せて調査). `loftr_predict` の fp16 化等は定常 median には効くが裾と無関係のため低優先.

### loftr パス完走とプロファイル付きベンチの対称比較

プロファイル付き再ベンチの loftr パスが完走し, 両バックエンドの対称比較が揃った. wall は eloftr 683.1s (2.83fps) vs loftr 713.7s (2.71fps, 差 4.3%), total median 163.5 vs 178.7ms (8.5%), マッチャー段 (`loftr_predict`) median 61.8 vs 79.0ms (21.8%). 裾構造は両者同一 (nerf_wait が wall の 52.7% / 50.7%). マッチャー単体では eloftr が明確に速いが, Amdahl 希釈と NeRF 律速により全体差は小さい, という 07-03 の分析結論を対称データでも確認した.

### PERF_plan.md の再編

perf CSV 分析の確定を受けて `notes/PERF_plan.md` を書き直した (日本語技術文書の文章規範に従いパラグラフライティングで再構成, 独自ラベル体系は不使用). 変更量ベースの旧 Tier 分類を廃し, 次の構成にした: (1) 目標の定義 (新しい観測から計算した姿勢の出力頻度だけを成果と数え, 同一姿勢の再出力やフレーム間引きを除外) と, 修正ごとの確認方法 (確認不要 / 軌跡整合 / 絶対精度の 3 段階), (2) 改善項目を"NeRF の同期待ちをなくす""毎フレームの処理を削る""長時間運用で速度を保つ"の 3 系統に整理, (3) 実施順 8 ステップ (結果保存の非同期化 → ランナー再構築の廃止とウォームアップ排除 → VRAM 切り分け → キーフレーム走査上限 → 同期ポリシー見直し → マッチャー opt 設定 → 学習反復削減 → 保留分の再判断). 計測により旧 Tier 1 (C++ の同期とメモリ確保) は該当区間の実測が小さく大レバーではないと判明したため保留に格下げした. 旧 Tier 番号は新文書に併記し参照互換を維持. `notes/ISSUES.md` の性能項目も新構成への参照に更新した. HO3D ADD/ADD-S ベンチはユーザー判断で優先度を下げ中断 (バックグラウンドエージェントは HO3D_ROOT 書き換えと gdown 準備まで実施済み, `data_reader.py` の 1 行変更は作業ツリーに残置).
