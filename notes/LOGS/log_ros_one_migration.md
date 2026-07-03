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

### 改善項目1: NeRF warm-start の実装・効果確認・アブレーション

`notes/PERF_plan.md` 改善項目1 のうち"学習反復数の削減, または前ラウンドの重みの引き継ぎ"を実装した.
`config.yml` に `nerf_reuse_weights: True` と `n_step_warm: 300` を追加した.
`bundlesdf.py:385` で NeRF の継続学習に `reuse_weights=cfg_nerf.get('nerf_reuse_weights', True)` を渡し, 毎ラウンドのモデル乱数初期化 (`create_nerf`) をスキップして前ラウンドの重みを保持する warm-start にした.
`bundlesdf.py:391-395` と `nerf_runner.py` の `train()` に `n_iters` 引数を追加し, 2 ラウンド目以降 (warm ラウンド) の学習反復数を 501 から `n_step_warm=300` に削減した.
`bundlesdf.py:762-767` の NeRF 同期待ちループに子プロセスの死活チェックを追加した. これは `notes/REVIEW_findings.md` の #7 (NeRF 子プロセスが死んでも無限に待つ) の修正であり, 子プロセスが死んでいれば `RuntimeError` を送出する.
`nerf_runner.py:388,396` では `reuse_weights=True` 経路の潜在デバイスバグを修正した. `self.rays` はラウンド末尾で CPU に移動するため, `pose_array`/`feature_array` の移動先を `self.rays.device` から `self.c2w_array.device` に変更した. この経路は従来未使用だったため, バグは今回の実装で初めて露見した (3 ラウンド目でデバイス不一致クラッシュを起こしていた).

フルベンチ (eloftr, 1932 フレーム) で効果を確認した.
壁時計は 683.1s から 501.4s に短縮した (−26.6%).
実効 fps は 2.83 から 3.85 に向上した (+36.2%).
`nerf_wait` (NeRF 同期待ちの壁時計占有率) は 52.7% から 35.9% に低下した.
NeRF ラウンドの median は 6.11s から 4.34s に短縮した.
フレーム median は 163.5ms から 161.6ms とほぼ不変であり, 定常経路 (毎フレームの姿勢計算) は無変更のままである.
`ob_in_cam` の出力数は 1932 個で, 連続フレームで同一姿勢が繰り返し出力された箇所は 0 件だった. 出力レートの数字だけを上げる見せかけの改善ではないことを確認した.

軌跡整合の確認では, baseline との比較で回転差 median が 1.885° となり, `notes/PERF_plan.md` の基準 (eloftr/loftr 間の回転差 median 1.33°) を超過した (超過率 71.7%). 並進差は median 0.160cm で基準ぎりぎり通過した.

学習反復数の削減と重み引き継ぎのどちらが速度改善の原因かを切り分けるため, アブレーションを実施した.
`n_step_warm` を 501 (削減なし) に固定し, 重み引き継ぎのみを有効にした条件では, 壁時計 718.7s (baseline 比 +5.2%, 改善が消失), 回転差 median 1.462° (基準をわずかに超過) となった.
この結果から, 速度改善はほぼ全部が学習反復数の削減 (501→300) 由来であり, 重み引き継ぎ単体は速度にほぼ寄与せず, 精度には副次的な残差 (基準超過分の約 1/4 に相当) を持つと判断した.

`runner_build` (毎ラウンドのモデル・オプティマイザ再構築) についても再計測した.
`notes/PERF_plan.md` はスモークベンチ (9 ラウンド) の計測に基づき, ランナー再構築の廃止を"無害な最適化"として見込んでいた.
しかしフルスケール (111 ラウンド, octree/ray のサイズが大きい) では median 793ms から 750ms への低下にとどまり, 見込んでいた効果はほとんど消えることが分かった.

終盤 (idx 1600-1890) に回転差が 10° を超えるスパイクが集中して現れた.
これが warm-start に起因するかを切り分けるため, マッチャーのみを入れ替えた eloftr-vs-loftr の対照比較を確認したところ, 同じ区間に同様のバーストが現れた.
このことから, バーストの原因は入力側 (低テクスチャ・対称形状のミルクジャグ, モーションブラー) にあり warm-start とは無関係と判断し, 判断材料から除外した.

以上を踏まえ, 現状の設定 (`reuse_weights=True`, `n_step_warm=300`) を暫定採用として作業ツリーに残置した (未コミット).
既定として確定するには HO3D データセットでの ADD/ADD-S による絶対精度確認が必要であり, 相対軌跡整合はあくまで代理指標にすぎないと判断したためである.

### サブエージェントのモデル自己申告の食い違い調査

Agent ツールで `model` パラメータを明示指定 (opus/sonnet/haiku) して子エージェントを起動したところ, 子エージェントが自己申告で親モデル名 (Fable 5) を名乗る事象が見られた.
実際にどのモデルが動いていたかをトランスクリプトファイル (`subagents/agent-*.jsonl`) の assistant メッセージの `message.model` フィールドで確認したところ, 指定どおりのモデル (opus→`claude-opus-4-8`, sonnet→`claude-sonnet-5`, haiku→`claude-haiku-4-5`) が記録されており, `model` オーバーライドは API レベルで正しく機能していることを確認した.
原因はシステムプロンプト内にモデル名の記載が 2 箇所あり, 内容が矛盾していたことにある. Environment セクションの記載は stale な親モデル名のままで, 後半のブロックには正しい子モデル名が入っていた.
子エージェントの自己申告は前者を参照していたため, 実際の稼働モデルと食い違って見えた.
この事象は claude-code の既知報告 (Issue #64848, model self-identification stale) と整合する.
教訓として, サブエージェントの稼働モデルを検証する際は自己申告ではなくトランスクリプトの `message.model` を参照することとした.

### PERF_plan.md の書式改定 (実施状況の可視化)

ユーザーから, `notes/PERF_plan.md` を TODO のように各項目の実施・未実施が一目でわかる形式にしてほしいとの指示を受けた.
改善項目 1/2/3 と実施順の各箇条書きに `[x]` (実施済み) / `[ ]` (未着手) のチェックボックスを追加した.
実施順セクションはチェック状態の一次情報を改善項目側に持たせ, 実施順側は優先順位のビューとして同期させる設計にした.

チェックボックス追加後, ユーザーから改善項目 1 の一項目 (学習反復数の削減, または前ラウンドの重みの引き継ぎ) について, 改善案を書くセクションなのか成果を書くセクションなのか判別できないとの指摘を受けた.
原因は, 施策の説明・実測結果・採用判断が地の文でひと続きになっており, 読み手がどのモードで読むべきか構造から判別できない点にあった.
該当項目のみ"実装""結果""判断"の 3 つの小見出しに分離して修正した.
他の実施済み項目 (ランナー再構築の廃止, NeRF 子プロセス死活チェック追加) は元々短く自然に読めるため変更していない.

### HO3D ベンチマークデータの取得

readme.md の "Data download" 節に従い, Google Drive から `evaluation.zip` (6.3GB, HO3D augmented eval-split の rgb/depth/meta/GT 動画データ), `masks_XMem.zip` (30MB, 物体マスク事前計算済み), YCB-Video 物体モデル zip (385MB) の 3 点を `data/ho3d_dl/` にダウンロードした. `masks_XMem` と `models` は全展開し, `evaluation.zip` は `SM1` (マスタードボトル, 898 フレーム) のみを `data/HO3D_v3/{evaluation/SM1,masks_XMem,models}` に展開した (残り 12 動画は zip 内に残置し, 必要になれば個別に追加展開できる). `BundleTrack/scripts/data_reader.py` の `Ho3dReader` がこのレイアウトから K/depth/mask/GT 姿勢/GT メッシュを正しく読み込めることをコード変更なしで確認した (`HO3D_ROOT` は前セッションから既に `/workspace/data/HO3D_v3` を指すよう設定済みだった).

なお, `notes/PERF_plan.md` の改善項目のうち"確認不要"に分類された項目群 (`save_result` 非同期化, `loftr_wrapper.py` の転送順修正, 二重 `no_grad` 削除, `confs_gpu` の cudaFree 漏れ修正, `astype(bool, copy=False)`, f-string ログの遅延評価, 対応点分割のソートベース化) はサブエージェント経由で並行実装した. 詳細は `notes/PERF_plan.md`/`notes/ISSUES.md`/`notes/REVIEW_findings.md` を参照 (いずれも当該セッション内で更新済み).

### 精度ガードレール構築: run_ho3d.py / benchmark_ho3d.py の完走とバグ修正

`run_ho3d.py --video_dirs .../SM1` を現行既定設定 (eloftr バックエンド + warm-start `reuse_weights=True`/`n_step_warm=300`) で実行し, 898 フレーム全てでエラーなく完走, `ob_in_cam/*.txt` が全フレーム分出力されることを確認した.

続けて `benchmark_ho3d.py` を実行したところ, 本フォークの ROS/perf 作業とは無関係な upstream 由来のバグを 2 件踏んだ. 1 件目は `benchmark_ho3d.py:156` で, `argparse.parse_args()` 直後に到達不能な `args = []` という行があり, `benchmark_one_video()` (グローバル変数として `args.out_dir`/`args.log_dir` を参照する) をループ呼び出しする直前で `args` を空リストに上書きしてしまい, 即座に `AttributeError: 'list' object has no attribute 'out_dir'` で落ちていた. 該当行を削除して修正した. 2 件目は `Utils.py` の `trimesh_clean()` で, インストール済み `trimesh` 4.12.2 で既に削除されている `remove_degenerate_faces()`/`remove_duplicate_faces()` を呼んでおり `AttributeError` になっていた. 現行の mask ベース API である `mesh.update_faces(mesh.nondegenerate_faces())` / `mesh.update_faces(mesh.unique_faces())` に置き換えて修正した (`remove_infinite_values()`/`remove_unreferenced_vertices()` は 4.12.2 でも有効なため変更していない). いずれも 1 行規模の trivial な修正でありサブエージェントを介さず直接対応した. 両修正とも作業ツリーに残置し未コミット.

修正後 `benchmark_ho3d.py` は完走し, SM1 (マスタードボトル, 898 フレーム, eloftr + warm-start) で ADD 2.20cm, ADD-S 0.98cm, ADD_AUC 78.10%, ADDS_AUC 90.18%, chamfer distance 0.52cm を得た. 出力一式は `data/bench_results/ho3d_log/` と `data/bench_results/ho3d_ours/SM1/` (いずれも gitignore 対象). この数値自体が良好かどうかは upstream 論文値との比較が必要でまだ評価しておらず, 本セッションの主眼はそこではなく, GT 付き精度ガードレール (`run_ho3d.py` → `benchmark_ho3d.py`) が一気通貫で動く状態を作ったことにある. 今後の絶対精度比較 (`BUNDLESDF_MATCHER=loftr` での再実行による eloftr との比較, warm-start on/off の比較, その他 `notes/PERF_plan.md` の"絶対精度の確認"段階の項目) は, この枠組みを設定だけ変えて再実行すれば着手できる.

### bench_milk.sh フルベンチによる 3 件の効果測定 (save_result 非同期化・転送順修正・confs_gpu リーク修正)

eloftr フルベンチ (1932 フレーム) を `data/out_milk_eloftr_async/` に完走させ, warm-start 適用済みの直前 baseline `out_milk_eloftr_item1` (壁時計 501.4s) と比較した. 壁時計は 683.1s (orig) → 501.4s (item1, warm-start) → 468.3s (async, 今回 3 件込み, item1 比 -6.6%) となり, 実効 fps は 2.83 → 3.85 → 4.13 に上がった. フレーム total median は 163.6ms → 161.7ms → 139.3ms, p90 は 227.9ms → 224.5ms → 198.6ms, nerf_wait は 359.9s (52.7%) → 180.1s (35.9%) → 193.5s (41.3%) だった.

3 件の内訳を切り分けると, 効果の大部分は結果保存の非同期化に由来する. `save_result` median は 21.95ms (orig) → 21.99ms (item1) → 0.01ms (async) まで下がり, これがフレーム median の -22.4ms と壁時計 -6.6% にほぼ対応する. `PERF_plan.md` の見込みどおりの効果だった. 転送順の修正 (`loftr_wrapper.py:80-81` の `.cuda()`/`.float()` 順序変更) は `loftr_predict` median を 61.83ms (item1) → 58.98ms (async) へ -3.0ms 縮め, 見込み (「数 ms」) と一致した. `confs_gpu` の cudaFree 漏れ修正 (`FeatureManager.cpp:1712`) については, VRAM 推移が 15.6→16.2→16.8→20.6→27.4GB (ピーク 27.7GB) と baseline (17→26GB) と同型の単調増加を継続しており, マクロな VRAM 曲線には効いていないことが分かった. 修正が対象とする確保・解放は RANSAC ペア単位の小さい規模であり, VRAM 単調増加の主因はキーフレーム蓄積 (終盤 262 keyframes) にあると判断した. ただしこの判断は後述する簡易計測に基づくものであり, 正式な VRAM ロギングでの確証はまだ得ていない.

見せかけ高速化の懸念に対しては, 新規作成した `scripts/check_repeated_poses.py` (未コミット) で `ob_in_cam` 1932 個を検査し, 連続フレーム同一姿勢の繰り返しが 0 件, 並進 median 0.178cm・回転 median 1.05° であることを確認した. フレーム間引きによる偽の高速化ではない.

### VRAM 計測ロジックの実装と, 今回の計測が正式実装によらないことについて

`notes/PERF_plan.md` の「VRAM 単調増加の要因確認」は, 既存のベンチコードに VRAM 計測が一切なく実施不能だったため, `perf_logger.py` に `_vram_mib()` を追加し, `SpanProfiler.flush()` 時に `vram_alloc_mib` (`torch.cuda.memory_allocated`) と `vram_reserved_mib` (`torch.cuda.memory_reserved`) を CSV 末尾に追記するようにした (`BUNDLESDF_PROFILE=1` 時のみ有効, CUDA 不可時は nan を返し例外を出さない. `perf_main.csv`/`perf_nerf.csv`/`perf_nerf_train.csv` 全てに適用, 既存の timing 列は無変更). `scripts/bench_milk.sh` にも, `nvidia-smi --query-gpu=timestamp,memory.used,memory.total -l 5` を単一プロセスでバックグラウンド起動する `start_gpu_mon`/`stop_gpu_mon` を追加し, `trap EXIT INT TERM` で確実に停止させ, プロセス全体 (メイントラッキング+NeRF ワーカーの合算) の VRAM 推移を `${BACKEND}_gpu_mem.csv` に記録するようにした. 最初は while+sleep によるループでポーリングする実装にしたが, 停止時にゾンビプロセスが残ることを実測で確認したため, 単一の `nvidia-smi -l 5` プロセスを起動する方式に変更した経緯がある. smoke test (256MiB 確保後に flush) で `vram_alloc_mib=256.0` 等の妥当な値が出ることを確認し, 既存の集計ツール `scripts/perf_stats.py` が新しい列があってもクラッシュしないことも確認した. 両ファイルとも未コミットである.

この実装には時系列上の注意点がある. 上記の bench_milk.sh フルベンチ (async) は, この VRAM 計測ロジックの実装が完成する前後のタイミングで実行されており, 実際に `data/out_milk_eloftr_async/perf_main.csv` を確認すると `vram_alloc_mib`/`vram_reserved_mib` 列は入っていない. そのため今回の VRAM 切り分けの数値根拠は, この正式実装ではなく, ベンチ実行時に別途用意した簡易計測 (`data/bench_results/async_vram.csv`, timestamp と memory_used の 2 列のみ) によるものである. 次回セッションでは, このロジックがフルベンチ実行時に実際に `perf_main.csv` へ列を出力することの動作確認がまだ済んでいない.

### gridencoder の ModuleNotFoundError

bench-verify のフルベンチ実行前, frame 116 (初回 NeRF ラウンド) で NeRF ワーカーが `ModuleNotFoundError: No module named 'gridencoder'` でクラッシュした. 2026-07-03 に追加済みの NeRF 子プロセス死活チェック (`bundlesdf.py:762-767`, PERF_plan.md 改善項目1 参照) がこれを検知して処理を中断しており, 死活チェック自体は意図どおりに機能した.

原因は `mycuda/setup.py` の `gridencoder` 拡張がトップレベルのモジュールとしてビルドされ, `.so` が `/workspace/mycuda/gridencoder.cpython-310-*.so` に配置される一方, `mycuda/torch_ngp_grid_encoder/grid.py:23` は自ディレクトリのみを `sys.path` に足して bare `import gridencoder` していることにある. NeRF ワーカーは `multiprocessing.Process` で spawn され, その `sys.path` は repo root のみを含むため, `gridencoder` を解決できない. 7/2 時点の baseline ではこのエラーが出ていなかったが, これは当時 `.so` が別の場所にあったためで, 直近のリビルドで配置パスが変わったことが原因と考えられる.

今回は config・コードを変更せず, NeRF ワーカー起動時の環境変数に `PYTHONPATH=/workspace/mycuda` を追加してその場しのぎで回避したのみである. これは NeRF を回す全ての実行に影響する環境問題であり, `grid.py` が親ディレクトリを `sys.path` に足すように直すか, `gridencoder` を import 可能な場所に install するかたちの恒久対処が必要で, 次回セッションの最優先課題とする.

### eloftr vs loftr の HO3D SM1 ADD/ADD-S 比較 (既定バックエンド確定)

`BUNDLESDF_MATCHER=loftr PYTHONPATH=/workspace/mycuda python3 run_ho3d.py --video_dirs data/HO3D_v3/evaluation/SM1 --out_dir data/bench_results/ho3d_ours_loftr` を実行し, HO3D SM1 (マスタードボトル, 898 フレーム) を loftr バックエンドで完走させた. 898/898 フレームで `ob_in_cam/*.txt` を出力し, gridencoder エラー・Traceback・CUDA error・Killed のいずれもゼロだった. 壁時計は約 364.6 秒 (03:13:21.92 開始→03:19:26.47 終了, 実測). `config.yml`/コードは無変更で, 環境変数 `PYTHONPATH=/workspace/mycuda` のみで gridencoder 問題 (既知の環境問題, `notes/ISSUES.md` 記載) を回避した. eloftr の既存結果は上書きせず, 別ディレクトリ (`data/bench_results/ho3d_ours_loftr/`, `data/bench_results/ho3d_log_loftr/`) に分離して保存した.

`benchmark_ho3d.py` で生 pkl から再計算し検証した精度比較は次のとおり (両者とも n=895, 898 フレーム中 3 フレームは評価対象外. eloftr/loftr 共通仕様のため比較の公平性に影響しない. 同一 warm-start 設定・同一 seed=0 で条件は揃えている).

| 指標 | eloftr (既定, 現行) | loftr (旧) | 差 |
|---|---|---|---|
| ADD mean (cm) ↓ | 2.20 (2.1954) | 2.77 (2.7671) | +0.57cm (loftr 悪化, +26%) |
| ADD-S mean (cm) ↓ | 0.98 (0.9833) | 1.21 (1.2083) | +0.23cm (loftr 悪化, +23%) |
| ADD_AUC (%) ↑ | 78.10 (78.099) | 72.39 (72.389) | -5.71pt (loftr 悪化) |
| ADDS_AUC (%) ↑ | 90.18 (90.184) | 87.94 (87.939) | -2.24pt (loftr 悪化) |
| chamfer (cm) | 0.52 (0.518) | 0.525 | +0.005cm (実質同等, NeRF 由来でバックエンド非依存のため妥当) |

姿勢精度 (ADD/ADD-S/AUC) の 4 指標すべてで eloftr が明確に優位だった. ADD_AUC で 5.7 ポイント, ADD 平均で 26% の差があり, 898 フレーム規模のサンプルでは誤差の範囲を超える有意差と判断した. 既定バックエンドを eloftr とする根拠が得られ, `notes/PERF_plan.md` の「既定採用の確定は精度比較 (HO3D の ADD/ADD-S) を待つ」というゲート条件は満たされた. これにより `notes/PERF_plan.md` 改善項目 2 の「マッチャーの opt 設定への切替」(現状 eloftr 前提で議論されている項目) の前提も確定した. なお, これは SM1 単一動画・単一 seed での比較であり, 他動画での再現性までは確認していない.

実行中に判明した副次的な事象として, プロセス完了判定に `kill -0 <PID>` を使ったところ, プロセス終了後もゾンビ状態 (`<defunct>`) の間は `kill -0` が真を返し続け, 完了検知が遅延する事象が発生した (実害なし, 監視方法の教訓として記録).

生データ保存場所: loftr 姿勢出力 `data/bench_results/ho3d_ours_loftr/SM1/ob_in_cam/`, loftr benchmark 出力 `data/bench_results/ho3d_log_loftr/` (ho3d_ours.xlsx/.pkl, pred_mesh*, gt/pred ply), loftr 実行ログ `data/bench_results/loftr_bench_run.log`, eloftr 既存結果 (今回変更なし) `data/bench_results/ho3d_ours/SM1/`, `data/bench_results/ho3d_log/`.

## 2026-07-03 (続き): 既知バグ3件の修正と VRAM 計測ロジックのフルベンチ動作確認

### gridencoder ModuleNotFoundError の恒久対処

前節で記録した gridencoder の `ModuleNotFoundError` を恒久対処した. `mycuda/torch_ngp_grid_encoder/grid.py:14` に `sys.path.append(os.path.dirname(code_dir))` を 1 行追加し, `gridencoder.so` の実配置場所である親ディレクトリ `mycuda/` を `sys.path` に加えた. 既存の `sys.path.append(code_dir)` はそのまま残置している. Python ファイルのみの変更であり再ビルドは不要だった.

検証には, `min_rot`/`start_nerf_keyframes` を一時的に下げて NeRF ラウンドを早期に発火させる 80 フレームのスモークテストを用いた. NeRF ワーカー (`multiprocessing.Process` で spawn) が実際に起動し, `PYTHONPATH` ハックなしで `GridEncoder` のロードに成功し, 複数ラウンドが完走することを確認した. これにより, これまで NeRF を回す全実行に必要だった `PYTHONPATH=/workspace/mycuda` によるその場しのぎの回避策は不要になった. git diff で変更内容を確認済みだが未コミット.

### REVIEW_findings #5 (percentile 空配列クラッシュ) の修正

`bundlesdf.py:757-762` に `if valid.any():` ガードを追加した. 完全遮蔽やフレームアウトで `valid` 配列が全 False になったとき, `np.percentile(空配列)` が `IndexError` で落ちる問題を修正するもので, 有効画素がないときは閾値計算と denoise をスキップしてログ出力のみ行うようにした. denoise は「遠すぎる深度を 0 にする」処理であり, 有効画素がゼロならそもそも denoise 対象も存在しないため, スキップが妥当な挙動と判断した. 最小再現で, 旧コードは `IndexError` で落ち, 新コードは安全にスキップすることを確認した. Python のみの変更で再ビルド不要. git diff で確認済みだが未コミット.

### REVIEW_findings #8 (FAIL 非 return) の修正

`BundleTrack/src/Bundler.cpp:901` (`Bundler::optimizeGPU` 内) で, `global_corres.size()==0` のとき FAIL フラグを立てた直後に `return;` を追加した. 修正前は FAIL を立てても処理を続行し, 対応点ゼロのまま最適化した無意味な結果を無条件に書き戻していた. 呼び出し元 (`bundlesdf.py:721` 付近) は `_status==FAIL` を見て forgetFrame + return するため, 早期 return を追加しても後続処理への悪影響はない. C++ の変更のため `bash build.sh` で `my_cpp` 拡張を再ビルドし, 警告のみでエラーがないことを確認した. `global_corres=0` の意図的な再現は困難なため, 80 フレームのスモークベンチで正常系 (optimizeGPU 経路が多数回実行される通常のトラッキング) が壊れていないことを確認して代替とした. git diff で確認済みだが未コミット.

以上 3 件の変更対象ファイルは `mycuda/torch_ngp_grid_encoder/grid.py`, `bundlesdf.py`, `BundleTrack/src/Bundler.cpp` の 3 つで, いずれも未コミットのまま作業ツリーに残置している.

### VRAM 計測ロジックのフルベンチ動作確認

前節で実装した VRAM 計測ロジック (`perf_logger.py`/`bench_milk.sh`) について, 「smoke test のみ確認済みでフルベンチでの動作確認は未実施」だった残課題を解消した.

`BUNDLESDF_PROFILE=1 BUNDLESDF_MATCHER=eloftr scripts/bench_milk.py --max_frames 30` を実行し (出力先 `data/out_milk_vramcheck/`, 既存のフルベンチ出力とは別ディレクトリ), `perf_main.csv` の末尾に `vram_alloc_mib`/`vram_reserved_mib` 列が実際に出力されることを確認した. 値は 1 フレーム目が alloc 314.3 / reserved 534.0 MiB, 定常が alloc 322.5 / reserved 954.0 MiB で, alloc はほぼ一定でリークの兆候はなく, reserved はキャッシュアロケータのウォームアップ後にプラトーに達する妥当な挙動だった. `perf_nerf.csv` もヘッダに両列が正しく含まれることを確認したが, 30 フレームではキーフレーム蓄積が 1 個のみで NeRF ラウンド自体が発火しておらず, データ行での実値確認はできていない (列スキーマの正しさのみ確認済み).

`scripts/bench_milk.sh` の `start_gpu_mon`/`stop_gpu_mon` 機構についても, 12 秒の直接実行で `eloftr_gpu_mem.csv` が想定どおりの形式 (timestamp, memory.used, memory.total) で出力され, 停止後に nvidia-smi のゾンビプロセスが残留しないことを確認した. 既存の集計ツール `scripts/perf_stats.py` が新しい CSV を読み込んでもエラーを出さないことも確認した. 本節の作業はいずれも動作確認のみでコード変更はなく, コミットもしていない.
