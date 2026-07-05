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

## 2026-07-03 (続き): キーフレーム走査の上限設定 (batched GPU covisibility)

`notes/PERF_plan.md` 改善項目2「キーフレーム走査の上限設定」(`Bundler.cpp:298,509`) に着手した. キーフレーム選択処理 (`select_kf`) は新フレームごとに既存の全キーフレームとの共視性 (covisibility) を計算する構造のため, キーフレーム数の増加に伴い O(N²) 的に劣化していた (実測: 序盤 2.6ms→終盤 24.9ms).

### 検討した3案

1. **走査に上限を設ける案 (却下)**: 直近 N 件のキーフレームだけを走査対象にする最も単純な案. ミルク動画は回転を伴う撮影であり, 古いキーフレームでも共視性が高くなり得る. 直近 N 件に絞ると BA (バンドル調整) に使うフレーム集合が変わり, 軌跡精度に影響するリスクがあると判断し却下した.
2. **naive GPU 版 covisibility 復活案 (却下)**: `Bundler.cpp` にコメントアウトされていた単発 GPU 呼び出し (`computeCovisibilityCuda`) を, キーフレームごとに 1 回ずつ呼ぶ方式. 400 フレームのスモークで実測したところ, キーフレーム 1 件あたりのメモリ確保・解放・同期のオーバーヘッドが支配的になり, CPU 版 (7.75ms) より遅い 19.2ms という結果になり却下した. upstream 作者がコメントアウトしていた理由もこれと推測される.
3. **batched GPU 版 covisibility (採用)**: 共視性計算をキーフレームごとに個別に呼ぶのではなく, 全キーフレーム分をまとめて 1 回の GPU カーネル起動 (grid.z=n_kf) で計算する方式. 計算式自体 (法線と eye ベクトルの内積, 閾値判定) は CPU 版・naive GPU 版と同一であり, 呼び出し回数だけを 1 フレームあたり 1 回に集約してオーバーヘッドを償却した.

### 実装

3 ファイル, +84 行程度の変更で, いずれも未コミットである.

- `BundleTrack/src/cuda/CUDAImageUtil.cu`: `computeCovisibilityBatchKernel` (CUDA カーネル) と `computeCovisibilityBatch` (host 関数) を新規追加した. カーネル完了後に `cudaMemcpy` (DeviceToHost, 非 async) で結果を回収しており, これが暗黙的な同期点になっている.
- `BundleTrack/src/cuda/CUDAImageUtil.h`: 上記の宣言を追加した.
- `BundleTrack/src/Bundler.cpp:506-517`: `selectKeyFramesForBA()` 内の `normal_orientation_nearest` 分岐で, OMP 並列 CPU ループを, pose 配列構築 + 1 回の batched GPU 呼び出しに置換した.

ROI の扱いについて, CPU 版・naive GPU 版は `fA->_roi` で走査範囲を絞っていたが, batched 版は ROI 引数を受け取らず画像全体を走査する. ただし `bundlesdf.py:547` で `roi = [0,W-1,0,H-1]` と常に ROI が画像全体に設定されているため, 実質的な計算範囲は変わらないことをコード確認により検証した. stride (2) と計算式も CPU 版と一致することを確認した. `bash build.sh` でリビルド成功, `my_cpp` import も確認済みである.

### 検証結果

フル 1932 フレーム, `data/out_milk_eloftr_kfcap/` (kfcap) を baseline `data/out_milk_eloftr_async/` (async) と比較した. 軌跡差は `scratchpad/kfcap_vs_async.csv` の生 CSV から再計算し裏取りした.

軌跡整合ゲートは合格した. 回転差 median 0.587° (基準 1.33° 未満), 並進差 median 0.064cm (基準 0.16cm 未満). p90 は回転 3.32°/並進 0.55cm, max は回転 31.9°/並進 2.34cm だった. rot>5° の 168 フレームは大半 (162 件) が idx 1576-1879 に集中しており, これは `notes/PERF_plan.md` 記載の既知の終盤区間 (低テクスチャ・対称形状のミルクジャグ, モーションブラー起因) であり, eloftr-vs-loftr 対照でも同区間にスパイクが出現する既知の現象であって, 本変更起因ではないと判断した.

select_kf は劇的に改善した. 終盤 200 フレームの median は 23.9ms→0.578ms (約41倍), セッション合計は 32.48s→1.27s (-31.2s) だった. 区間ごとの推移 (0.012→0.567→0.644→0.578ms, キーフレーム 0→262 件) はほぼ横ばいであり, O(N²) 依存が解消されキーフレーム数に依存しなくなったことを示す.

フレーム総時間の median は 139.3ms→129.3ms (-10ms), 終盤 200 フレームでは 155.7ms→138.5ms (-17ms) であり, 定常計算時間は確実に短縮した. 一方で壁時計は不変だった (kfcap 501s, total_ms 合計 469.9s, vs async 468.3s). これは NeRF 同期待ちが壁時計の 52% を占め律速しているためであり (`notes/PERF_plan.md` の既存分析と整合), select_kf 単体の高速化が壁時計に反映されない.

VRAM はピーク 27.2GB で baseline (27.7GB) と同等だった. この変更は VRAM 単調増加の主因 (キーフレーム蓄積) には無関係であり, 影響なしという予想どおりの結果である.

また, `computeCovisibilityBatch` 内の `cudaMemcpy` (DeviceToHost) がカーネル完了を待つ暗黙の同期点になっているため, select_kf スパンの計測値 (0.578ms 等) は GPU 計算完了込みの実時間であることをコード確認により裏取りした (GPU 非同期実行による計測誤差の懸念はない).

### 生データ保存場所

フル出力 `data/out_milk_eloftr_kfcap/` (poses, perf_main.csv 等), `data/bench_results/kfcap_frame_times.csv`, `data/bench_results/kfcap_gpu_mem.csv`, 軌跡差 per-frame `scratchpad/kfcap_vs_async.csv`. baseline `data/out_milk_eloftr_async/` は読み取りのみで未変更である.

## 2026-07-03 (続き): 同期ポリシー見直し (sync_max_delay 拡大) の検証と不採用判断

`notes/PERF_plan.md` 改善項目1「同期ポリシーの見直し」(`config.yml:102` の `sync_max_delay` 拡大, または待ちの非ブロック化) に着手した. 待ちの機構は `bundlesdf.py:761-772` にあり, キーフレームの先行数が `sync_max_delay` 以上でかつ NeRF ラウンド実行中はスピン待ちする. 現行既定は 3 (kfcap baseline).

### 検証方法

`sync_max_delay` を 3 (baseline) から 6, 10 に拡大し, ミルクベンチ (1932 フレーム) と HO3D SM1 (898 フレーム, n=895, 必須ゲート) の両方で検証した. `config.yml` は検証後に 3 へ戻し, git diff がない状態に復元済みである.

### 速度側の結果 (ミルクベンチ, eloftr)

| delay | nerf_wait | 壁時計 | fps |
|---|---|---|---|
| 3 (kfcap, baseline) | 217.7s (46.3%) | 469.9s | 4.11 |
| 6 | 65.8s (20.5%) | 321.2s | 6.02 |
| 10 | 17.6s (6.4%) | 275.1s | 7.02 |

フレーム total median は delay によらずほぼ一定 (~129-132ms) であり, トラッキング自体の計算内容は変わらず NeRF 待ちの頻度だけが変わることを確認した. 見せかけ高速化でないことも `scripts/check_repeated_poses.py` で確認済みである (bit-identical consecutive poses は sync6/sync10 とも 0 件, フレーム間差は async/kfcap と同水準).

### 精度側の結果 (1): ミルク軌跡整合 (kfcap 基準との相対差)

| delay | 回転差 median | 回転 p90 | 回転 max | 並進差 median |
|---|---|---|---|---|
| 軌跡整合ゲート基準 | 1.33°未満で合格 | — | — | 0.16cm未満で合格 |
| 6 | 2.56° | 6.79° | 14.76° | 0.275cm |
| 10 | 3.32° | 11.37° | 32.45° | 0.302cm |

いずれも基準超過で不合格だった.

### 精度側の結果 (2): HO3D SM1 絶対精度 (必須ゲート, n=895)

| delay | ADD(cm) | ADD-S(cm) | ADD_AUC(%) | ADDS_AUC(%) | chamfer(cm) |
|---|---|---|---|---|---|
| 3 (既定) | 2.20 | 0.98 | 78.10 | 90.18 | 0.52 |
| 6 | 2.63 | 1.16 | 73.78 | 88.38 | 0.52 |
| 10 | 2.74 | 1.18 | 72.67 | 88.20 | 0.51 |

delay 増加に伴い ADD/ADD-S が単調に悪化した (delay 3→6 で ADD +0.43cm/ADD_AUC -4.32pt, 3→10 で ADD +0.54cm/ADD_AUC -5.43pt). これは GT に対する絶対精度の劣化であり, 機構 (NeRF 姿勢補正の反映遅延) も説明できる. chamfer (形状復元) はほぼ不変であり, 姿勢劣化はメッシュ形状には及ばない.

### 判断: 不採用

sync_max_delay 拡大の既定採用は不採用と判断した. 速度向上は大きい (fps 4.11→7.02) が, HO3D 絶対精度が明確に悪化するため, warm-start (`notes/PERF_plan.md` 改善項目1の項目7, 軌跡整合ゲート不合格でも絶対精度未確認のまま許容された前例) とは異なり, 既定確定の基準 (絶対精度に明確な劣化がないこと) を満たさない. 速度優先でユーザーが許容する場合でも delay=6 を上限とすべきであり, delay=10 は収穫逓減 (精度劣化幅の割に速度向上が小さい) と判断した.

待ちの非ブロック化 (もう一方のアプローチ) は未実装のままである. 精度劣化の根本原因 (NeRF 補正の反映遅延) は非ブロック化しても残るため, 単純な非ブロック化では同じ問題を引き継ぐ可能性が高いと判断し, 次回以降の検討課題とした (改善項目3 の MPS 導入で NeRF スループット自体を上げる方が本質的な解決に近い).

生データ: `data/out_milk_eloftr_sync{6,10}/`, `data/bench_results/sync{6,10}_*`, `scratchpad/kfcap_vs_sync{6,10}.csv`, `data/bench_results/ho3d_ours_sync{6,10}/`, `data/bench_results/ho3d_log_sync{6,10}/`. 既存 baseline (kfcap, ho3d_ours, ho3d_log) は上書きしていない. 未コミット (`config.yml` は検証後に diff なしへ復元済みのため実質コミット不要).

### ユーザーとの議論: delay-精度トレードオフをどう決めるか

上記の検証結果を受けて, ユーザーと「sync_max_delay と精度のトレードオフをどう決めるか」を議論した.

このパイプラインの最終目的はロボット制御のための物体姿勢情報の提供であり, 精度の最大化自体が目的ではない. 「制御が成立する範囲の精度」であれば速度を優先すべきだが, その許容誤差の範囲は机上では決められず, 実機での検証でしか判断できない (ユーザー自身の言葉: 「制御が成立する範囲は実機で検証するしかない」). したがって, 実機検証で許容誤差の仕様が決まるまでの間に, delay と精度のトレードオフデータを先に蓄積しておき, 実機検証後に「この許容誤差ならこの delay」を逆引きできるようにしておく, という方針になった.

この方針転換の理由は, 机上の絶対精度ゲート (HO3D の ADD/ADD-S) だけでは delay の採否を一意に決められないと判明したためである. sync_max_delay 拡大は warm-start とは異なり絶対精度の明確な劣化を伴うため単純に「不採用」と判断できたが, 「では何 delay まで許容できるのか」は制御系の要求仕様 (許容誤差) 次第であり, 現時点ではその仕様自体が未定である. 機上のベンチだけで最適点を決めようとすると, 精度最大化 (delay=3 に固定) と速度最大化 (delay=10 を採用) のどちらの立場も正当化できてしまい, 判断が宙吊りになる. 実機検証を待たずに作業を止めるのではなく, 先にトレードオフデータを蓄積しておくことで, 実機検証が完了した時点で仕様に合う delay を即座に選べるようにする, という判断である.

### 新規タスク: delay-精度トレードオフの体系的スイープ

HO3D の評価セットは 13 動画あり, 内訳は次のとおりである (SM1 で使った 898 フレームより他は大幅に長い).

- AP10, AP11, AP12, AP13, AP14 (各 1616 フレーム)
- MPM10, MPM11, MPM12, MPM13, MPM14 (各 1618 フレーム)
- SB11, SB13 (各 1680 フレーム)
- SM1 (898 フレーム, 検証済み)

13 動画合計で約 20,428 フレームである. 全動画×細かい delay 刻みを一度に網羅すると計算コストが大きい (半日〜1日規模) ため, 段階的に進める方針で合意した.

1. **第1段階**: SM1 に加え, 物体・視点条件が異なる動画を 1〜2 本 (例: AP系 1本, MPM系 1本) 選び, delay を細かく振る (3, 4, 5, 6, 7, 8, 10, 12, 15 程度). 目的は「delay-誤差カーブの形状」(線形に劣化するのか, どこかで急に破綻するのか) を把握することである.
2. **第2段階**: 第1段階で見えた「精度が崩れ始める境界 delay」付近を中心に, 残りの動画にも広げて一般化性を確認する.
3. 深夜などロボット実機を使わない時間帯にバックグラウンドで回す運用を想定する.

まだ決まっていないこと (次回セッションで詰める): 第1段階で使う代表動画 (AP系/MPM系/SB系のどれを選ぶか), 実行方法 (このセッション内でバックグラウンド実行するか, `/schedule` 等で夜間に自動実行するか), 収集したデータの整理・可視化方法 (delay-誤差カーブとして後から参照しやすい形にする).

このタスクは `notes/TODO.md` の未完了項目として独立に追加した.

## 2026-07-03 (続き): notes の誤り訂正 (SAM3 統合イメージのビルド・import 確認は完了済みだった)

`notes/ISSUES.md`/`notes/TODO.md` には「SAM3 統合後イメージの再ビルドが未検証」「イメージの再ビルドと `Sam3VideoModel` の import 確認はユーザー指示で中断しており未実施」という記述が残っていたが, ユーザーの指摘を受けて実際のコンテナ環境を検証したところ, これは事実誤認だったと判明した.

検証したのは以下の項目である.

- `python3 -c "import transformers; print(transformers.__version__)"` → `5.12.1` (ISSUES.md 記載のバージョンと完全一致)
- `huggingface_hub`, `accelerate` もインストール済み
- OS: `Ubuntu 22.04 jammy` (統合コンテナの記述通り)
- `python3 -c "from transformers import Sam3VideoModel"` → 成功 (`<class 'transformers.models.sam3_video.modeling_sam3_video.Sam3VideoModel'>`)

つまり `ros-one.dockerfile` の統合イメージビルドと `Sam3VideoModel` の import 確認は, 実際にはすでに完了していた. ISSUES.md/TODO.md の「未実施」という記述が古いまま更新されずに残っていたことが原因である.

一方, 別件として SAM3 の重み (checkpoint/safetensors) はコンテナ内に見当たらず, 未配置のままであることも確認した (`find / -iname "*sam3*checkpoint*"` 等で該当なし). これは TODO.md の「SAM3 重み配置 (HF gated リポジトリ要承認申請)」が未完了のままであることと一致しており, こちらは引き続き別のブロッカーとして残る.

この訂正を受けて, `notes/TODO.md` の該当項目を `[x]` に変更し, `notes/ISSUES.md` の重複するエントリ (ビルド・import 確認の未検証扱い) を削除して TODO.md 側に一本化した.

## 2026-07-03 (続き): #6 mconf 空配列の修正

`notes/REVIEW_findings.md` #6 (`loftr_wrapper.py:118`, PLAUSIBLE 判定) を修正した. マッチ 0 件 (テクスチャ欠乏ペア) のとき `logging.info(f"mconf, {mconf.min()} {mconf.max()}")` の `mconf` が空配列になり, `ValueError: zero-size array to reduction operation minimum which has no identity` でクラッシュしていた.

修正は, 既存変数 `total_n_matches` (113 行目で計算済み) を使い, マッチが 0 件のときはこのログ行をスキップするガードを追加する形で行った.

```python
if total_n_matches > 0:
  logging.info(f"mconf, {mconf.min()} {mconf.max()}")
else:
  logging.info("mconf: no matches")
```

検証は, 無地 (テクスチャなし) のグレースケール画像ペア (480x640, 1 チャンネル) を `LoftrRunner.predict()` に渡すスモークテストで行い, マッチ 0 件でも `ValueError` を起こさず `corres[0].shape=(0,5)` を正常に返すことを確認した. また旧コードが確かに `ValueError` を起こすことも, `np.array([]).min()` により別途確認済みである. Python のみの変更で再ビルドは不要であり, 未コミットである.

副次的な発見 (修正不要, 記録のみ): テスト中に別の既存バグ (`loftr_wrapper.py:83` の `if image0.shape[-1]==3:` という grayscale 判定が, `.permute(0,3,1,2)` 後の呼び出しのため実質的に意味をなさない, `notes/REVIEW_findings.md` の「圏外の生存候補」に既に PLAUSIBLE として記載済みのバグ) を実地で踏んだ. 3 チャンネル RGB 画像を渡すと `RuntimeError: expected input to have 1 channels, but got 3 channels` になることを確認した. これは今回のスコープ外のため修正はしていないが, 実際に発火することが実証された点は記録に値する.

## 2026-07-03 (続き): loftr_wrapper.py:83 グレースケール判定バグの修正

上記の副次的発見を受け, `loftr_wrapper.py:83` の grayscale 判定バグも別途修正した.

原因は, `if image0.shape[-1]==3:` という入力 RGB 画像をグレースケールに変換すべきかどうかを判定するコードが, 直前の `.permute(0,3,1,2)` (81-82 行目, NHWC→NCHW 変換) の後に評価されていた点にある. これにより `shape[-1]` はチャンネル数ではなく幅 (W) を指しており, 実質的に意味をなさない判定になっていた.

調査の結果, 実運用 (`BundleTrack/config_ho3d.yml:10` の `USE_GRAY: true`) では C++ 側の `processImagePair` が Python 側に渡す前に既にグレースケール変換を行っているため, Python 側はこの判定に到達する時点で既に 1 チャンネル画像を受け取っており, このバグは実運用パスでは発火しない (常に False になるだけの休眠コード) ことを確認した. 一方, 3 チャンネル RGB 画像を直接渡すテストでは実際に `RuntimeError: expected input to have 1 channels, but got 3 channels` を引き起こすことは, 上記の mconf 修正のスモークテスト中に偶然踏んで確認済みである.

修正は `image0.shape[-1]==3` を `image0.shape[1]==3` (permute 後の NCHW 形式ではチャンネル軸は axis=1) に変更する 1 行で行った. 再ビルドは不要である.

検証は次の 2 点で行った.

1. 3 チャンネル RGB のテクスチャなし画像ペアを渡すテスト: 修正前は `RuntimeError`, 修正後は正しく grayscale 変換されエラーなく完走した (`corres[0].shape=(0,5)`, マッチ 0 件も正常処理).
2. 既存の実運用パス (1 チャンネルグレースケール入力, `USE_GRAY:true` 相当) が引き続き正常動作することも確認した (乱数画像で 4256 マッチを正しく検出, 挙動に変化なし).

未コミットである. なお, 実運用パスでは発火しないと判明したことから, このバグ自体の緊急性は低かった (次回セッションでの優先度判断の参考情報として記録する).

## 2026-07-03 (続き): notes の誤り再訂正 — SAM3 重み配置は完了済みだった

上記の SAM3 統合イメージ検証時, `find / -iname "*sam3*checkpoint*"` 等のファイル名検索で該当がなかったことから「SAM3 の重みは未配置」と判断し, TODO.md にもその旨を記載した。

ユーザーからの指摘 (「HF キャッシュにあるのでは?」) を受けて `$HF_HOME` (`/home/ak/.cache/huggingface`) を直接確認したところ, `hub/models--facebook--sam3/` に `model.safetensors` (3.3GB, 2026-05-07 取得) と `sam3.pt` が実在した。先の検索が失敗していたのは, ファイル名パターンの想定 (`*sam3*checkpoint*` 等) が誤っており, HF キャッシュの実際の構造 (ディレクトリ名にのみ `sam3` を含み, 実体ファイルは `model.safetensors` という汎用名で `blobs/` 配下にハッシュ名で格納される) を見落としていたためだった。

`python3 -c "from transformers import Sam3VideoModel; Sam3VideoModel.from_pretrained('facebook/sam3')"` を実行し, 1797 個の重みテンソルが 1.5 秒でロードされ (dtype float32), 実際に使用可能な状態であることを確認した。

これにより, `bundlesdf_node`/`sam3_segmenter` の実機検証に対するブロッカー (コンテナビルド, `Sam3VideoModel` import, 重み配置の3点) はすべて解消されていたことが判明した。TODO.md の該当項目を `[x]` に訂正した。

教訓: ファイルの存在確認は, 想定するファイル名パターンでの `find` だけに頼らず, 該当するキャッシュ/配置ディレクトリ (今回は `$HF_HOME`) を直接確認する方が確実である。

## 2026-07-03 (続き): MPS 導入後の全9改善項目の再評価

`notes/PERF_plan.md` の各改善項目 (ランナー再構築の廃止, warm-start, キーフレーム走査上限設定, 同期ポリシー, マッチャーの opt 設定切替, 同期ポリシーの非ブロック化, マッチャー呼び出しの統合, MPS 恒久化, 保留項目) について, MPS 有効/無効 (`data/out_milk_eloftr_kfcap/` vs `data/out_milk_eloftr_mps/`) の既存 perf CSV を再分析し, 判断が変わるか検証した. 新規 GPU 実行は行わず, 既存データの再分析のみである. 分析スクリプトは `scratchpad/analyze.py`/`analyze2.py` (未コミット, scratchpad のため git 対象外).

結論として, 判断が確定的に変わった項目はゼロだった (全ての既存の採否判断は MPS 下でも維持された). ただし優先度が変動した項目がある.

- **マッチャーの opt 設定切替は優先度が上昇した**. MPS が競合律速の処理 (ransac -73%) を削った結果, 計算律速のマッチャー推論 (`loftr_predict`, MPS では -7% 止まり) が定常フレームの 49.9% を占める最大チャンクになったためである.
- **マッチャー呼び出しの統合は優先度が低下した**. MPS がカーネル起動オーバーヘッドの競合を既に緩和しており, 統合で削れる固定オーバーヘッド自体が縮小したためである.
- **保留項目 (旧 Tier1: C++ 同期・メモリ確保等) は優先度が低下した**. ransac 比重が 10.1%→3.2% に激減し, これらの競合コストは既に MPS が吸収済みと判明したためである.
- **同期ポリシーの非ブロック化は判断変化なし**. `nerf_wait` 比率は 46.3%→45.1% とほぼ不変だった. MPS は NeRF ラウンドとトラッキング定常処理をほぼ比例して速くしたため待ち比率自体は動かず, 非ブロック化は `sync_max_delay` 拡大と同じ精度トレードオフを負うため, 判断は変わらない.
- **ランナー再構築の廃止は判断変化なし**. `runner_build` median は 901.9→803.7ms (-11%) で多少速くなったが, 大半が octree/ray 再構築という構造的コストで reuse 不可であり, 廃止しても効果薄という結論は不変である.

新たな重要発見として, 定常フレームの計算自体が MPS で約16ms速くなることが判明した (定常フレーム median 112.8ms 中. 以前は「select_kf 等の絶対値は MPS 有無で変わらない」と誤って前提していたが, これは誤りだった). 内訳: ransac 13.10→3.56ms (-73%), find_corres_local 67.52→53.54ms (-21%), loftr_predict 60.18→55.89ms (-7%), select_kf 0.57→0.39ms (-33%). 競合律速の処理ほど MPS の効きが大きく, 計算律速の処理では効きが小さいという非対称性があり, これがマッチャー opt 設定切替の優先度上昇の根拠になっている.

詳細な優先度変動と実施状況は `notes/PERF_plan.md`, 完了サマリーは `notes/TODO.md` を参照.

## 2026-07-03 (続き): delay-精度トレードオフスイープの完走と方針転換による中断

`notes/PLAN_delay_accuracy_sweep.md` の計画に基づき, AP10 (`pitcher_base`)・MPM10 (`potted_meat_can`)・SB11 (`bleach_cleanser`) の3動画 (いずれも MPS 有効環境) で確定グリッド delay=3, 4, 5, 6, 7, 8, 10, 15 の8点を実行し, 全24回のスイープを完走した.

結果, SM1 の sync_max_delay 検証で見られた「delay 拡大で単調に劣化する」パターンは3動画とも再現せず, 非単調な変動が見られた (AP10 では delay=3 が最悪, MPM10 では delay=15 が最良等). この非単調性が実行ノイズでなく再現性のある効果であることを確認するため, AP10 の delay=3, 4 を再実行したところ, 初回 ADD 2.900cm/1.696cm に対し再実行では 2.718cm/1.841cm となり, 方向性 (delay=3 の方が悪い) と大きさの両方が再現された.

この結果を受け, SM1 自体を MPS 有効環境・同一グリッドで再検証するタスク (`sm1-mps-sweep-verify`) を開始したが, ユーザー判断により中断した. 理由は, 3〜4動画のデータだけでは傾向 (単調/非単調) が確定できないため, いずれ13動画全部で網羅的なスイープを行う予定であり, それなら他の改善項目 (マッチャーの opt 設定切替等) を全部実装し終えた最終構成で1回のスイープにまとめる方が効率的という方針転換である.

中断作業中, バックグラウンドプロセスの停止に手間取る教訓を得た. Agent ツールで起動したエージェントを `TaskStop` で止めても, エージェントがバックグラウンドでデタッチして起動したシェルスクリプト本体は独立して生き残り続ける (PID1 の子として孤立する). 実際に `config.yml` が delay=10 まで書き換わり新しい `run_ho3d.py` が起動してしまったため, プロセスツリーを `ps -ef --forest` で辿って直接 kill する必要があった. 最終的に GPU・プロセス・`config.yml` ともにクリーンな状態に復元済みである (`config.yml` は `sync_max_delay=3`, git diff なし).

生データは保持している: `data/bench_results/ho3d_sweep/` (AP10/MPM10/SB11 各8delay点, 24回分の完全な delay-精度データ), `scratchpad/kfcap_vs_sync6.csv` 等. 方針転換の詳細と将来の網羅的スイープへの活用方針は `notes/PLAN_delay_accuracy_sweep.md` を参照.

## 2026-07-03 (続き): マッチャーの opt 設定切替の実施と採用

`notes/PERF_plan.md` 改善項目2「マッチャーの opt 設定への切替」を実施した. `loftr_wrapper.py` の `_init_eloftr` に環境変数 `BUNDLESDF_LOFTR_CFG` (値: full/opt) を追加し, EfficientLoFTR の `opt_default_cfg` (既存差分: `MATCH_COARSE.THR` 0.2→25, `SKIP_SOFTMAX` False→True, `FP16MATMUL` False→True) を使えるようにした.

検証 (MPS 有効環境, ミルクベンチ+HO3D SM1) の結果は次のとおりである.

- 軌跡整合: 回転差 median 0.930° (基準1.33°未満), 並進差 median 0.118cm (基準0.16cm未満) で合格.
- 速度: `loftr_predict_ms` 55.89→52.39ms (-6.3%). ただし `total_ms` は 112.80→116.43ms (+3.2%, NeRF 律速のため段単体の高速化が全体には反映されない).
- 絶対精度 (HO3D SM1, n=895): ADD 2.20→1.97cm, ADD-S 0.98→0.915cm, ADD_AUC 78.10→80.34%, ADDS_AUC 90.18→90.87%, chamfer 0.52→0.469cm. 全指標で opt 設定が full 設定を上回った (速度だけでなく精度も改善).

既定値を opt に変更して採用を決定し, コミット済み (f0702a5 "perf(matcher): default eloftr to opt config (skip_softmax + fp16matmul)"), push 済みである.

## 2026-07-03 (続き): SAM3 単体トラッキング vs SAM3+SOTAトラッキングの速度調査

`ros/sam3_segmenter/scripts/sam3_segmenter_node.py` が既に SAM3 の video tracking API (memory 機構, `init_video_session`/`add_text_prompt`/session 経由の frame 処理) を使っており, 毎フレーム独立推論ではないことを確認した.

外部情報調査 (一次情報で検証済み) の結果は次のとおりである.

- SAM3 (動画): 16fps (H100). 出典: Meta ブログ (https://ai.meta.com/blog/segment-anything-model-3/).
- SAM3.1 (動画): 32fps (H100, object multiplex で最大16物体/forward pass). 同上出典.
- Cutie-small: 45.5fps, Cutie-base: 36.4fps, XMem: 22.6fps, DeAOT-R50: 11.7fps. いずれも V100. 出典: Cutie 論文 Table 1 (ar5iv 版 https://ar5iv.labs.arxiv.org/html/2310.12982).

GPU 世代が揃っていない (Cutie 系は V100, SAM3 系は H100) ため単純比較はできないが, 桁感としては2段構成 (SAM3 で初期マスク→Cutie 等で伝播) の方が定常状態で数倍速いと見積もった.

## 2026-07-03 (続き): MPS 導入の恒久化 (実装のみ, Docker 実ビルドは未実施)

`notes/PERF_plan.md` 改善項目3「MPS の導入」は, セッション内での手動有効化による検証 (壁時計 -9.7%, fps +10.7% 等) は済んでいたが, `docker-compose.yml`/dockerfile への恒久組み込みは別タスクとして未着手のままだった. 今回, 恒久組み込みの実装を行った.

`docker/mps-entrypoint.sh` を新規作成した. 内容は次のとおりである. `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` を未設定時は `/run/nvidia-mps` にデフォルト設定する. `nvidia-cuda-mps-control -d` で MPS 制御デーモンを起動する (失敗しても非致命的にコンテナは起動継続させる). CMD をバックグラウンド実行して `wait` する. SIGTERM/SIGINT を trap し, CMD へシグナルを転送して終了を待ってから `echo quit | nvidia-cuda-mps-control` で MPS デーモンを確実に graceful shutdown する. `docker/ros-one.dockerfile` の末尾に `COPY docker/mps-entrypoint.sh`, `RUN chmod +x`, `ENTRYPOINT ["/usr/local/bin/mps-entrypoint.sh"]` を追加した. 既存の `CMD ["bash"]` は維持しており, entrypoint 経由でラップする形にした.

当初は exec で CMD を PID1 化しつつ trap でシグナルを処理する設計を想定していたが, exec はシェルプロセス自体を置換し trap を破棄するため両立しないと判明した. これを受けて標準的な「CMD をバックグラウンド実行して `wait` し, trap でシグナルを転送する」方式に変更した. これは docker の init ラッパーとして一般的なパターンである.

pipe dir (`/run/nvidia-mps`) はコンテナ内部パスであり, bind マウントされている `/tmp:/tmp` の外にあるため, ホスト側の MPS/CUDA プロセスと衝突しない設計である.

Docker の実ビルド・コンテナ再起動は現在の作業環境を壊すリスクがあるため実施していない. 今回行ったのは静的な構文チェックのみである.

`docker-compose.yml` への `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` 環境変数追加も本来必要だが, これは保留とした. 理由は, 同ファイルには直近コミット (`e243645`) 以降にユーザー自身の未コミット変更 (hf_cache volume 削除, `HF_HOME` 環境変数, Claude Code 関連マウント設定) が既に存在しており, 今回の MPS 変更が同じワーキングツリー上で混在してしまうためである. `docker-compose.yml` の扱いはユーザーが別途指示する.

## 2026-07-03 (続き): SAM3 セグメンターのノード分離を統合すべきかの検討 (結論: 保留)

ユーザーから「SAM3 を別ノードにしなくてよいのではないか. 別ノードにしたモチベーションは python バージョン違いによるコンテナ分離のためだったはずだが, 今はコンテナが統合されているし, BundleSDF のオリジナル実装ではセグメンター内蔵を意図していたはずだ」という提案があり検討した.

事実確認の結果, ユーザーの認識は正確だった. コンテナは `f8a0428 feat(docker): fold the SAM3 segmenter into the jammy image` で既に jammy 単一イメージに統合済みである (元々は python3.10 のバージョン競合により別コンテナに分離していた). オリジナルの `run_custom.py` は `segmenter = Segmenter()` を直接インスタンス化し `segmenter.run(...)` を同期呼び出しする「内蔵」設計だった. 一方, 現状は `ros/sam3_segmenter/scripts/sam3_segmenter_node.py` (RGB 受信→マスク publish) と `ros/bundlesdf_node/scripts/bundlesdf_node.py` (`message_filters` で rgb/depth/mask を time-synchronized subscribe, `slop=0.05`) という, ROS トピック経由の別ノード構成のままである.

検討したトレードオフは次のとおりである. 統合に慎重であるべき理由は, 直前のセッションで「SAM3 単体 vs SAM3+Cutie 等の軽量トラッカー」という構成変更の可能性を検討したばかりであり, ノードを1プロセスに統合するとセグメンテーションバックエンドの差し替え可能性 (疎結合性) が失われる点にある. ROS ノード分離が持つ障害分離のメリットも失われる. 統合を支持する理由は, `message_filters` の time-synchronized subscribe (`slop=0.05`) がタイムスタンプのずれによるフレーム取りこぼしのリスクを抱えており, この同期購読の仕組み自体がまだ実地で試されていない点にある. 内蔵化すればこの同期ずれ問題自体が構造的に消える. 折衷案として, 完全な1プロセス統合ではなく「同一プロセス内でセグメンターを関数呼び出しする (同期ずれ問題を解消する) が, セグメンテーション実装はインターフェースとして差し替え可能にしておく」という設計を提案した.

結論として, 実機検証がまだ行われていない段階での判断はリスクが高いため, まず一度実機で動かして実際に同期ずれが問題になるかどうかを見てから, 統合するかどうかを判断する方針とした. 今回は実装 (ノード統合) は行わず, 検討結果の記録のみである.

結論として, BundleSDF 全体は NeRF 同期待ちが壁時計の約50%を占め律速要因であり, セグメンテーション速度 (16fps程度でも十分) は現状ボトルネックになっていない. トラッカー切り替えによる速度差の実務的重要性は低いと判断した. 例外は VRAM 競合緩和の可能性があるが未検証である. 本調査は実装を伴わない調査のみである.

## 2026-07-04: REVIEW_findings.md #10 の修正と cleanup 系2件の対応

`notes/REVIEW_findings.md` #10 (`ros/sam3_segmenter/scripts/sam3_segmenter_node.py:38`, `init_video_session` の dtype 未指定) を修正した. 呼び出しに `dtype=self.dtype` を追加し, モデルの bf16 設定に対してセッション状態が fp32 固定になる不一致を解消した. 実機検証 (ROS 実機が必要) は未実施である.

cleanup 系の残り3件のうち2件を修正した. `nerf_runner.py:900-902` の `kpts_to_ray_ids` は, フレームマスクを一度 CPU へ転送してからそのマスクで `cur_rays` を再度インデックスしており GPU→CPU 転送が重複していた. マスクを GPU 上に保持したまま `cur_rays` をインデックスし, CPU への変換をその後の 1 回にまとめて削減した. `bundlesdf.py:651` の `matches = self.bundler._fm._matches[(frame, ref_frame)]` は代入後どこにも参照されないまま 658 行目で再代入されており, 死んだ代入だったため削除した.

残り1件 (`run_nerf`/`run_global_nerf` の約150行重複) は見送った. 一方はマルチプロセスワーカー関数, 他方はインスタンスメソッドであり実質的な分岐点があるため, 無理な統合はリスクが高いと判断した.

`notes/REVIEW_findings.md`/`notes/TODO.md` を更新済み. コミットはしていない.
## 2026-07-05: catkin ビルド検証, モックカメラストリーム実装, object_pose publish バグの FTA 修正

### catkin ビルドの実施・検証

`ros/bundlesdf_node`/`ros/sam3_segmenter` は catkin パッケージとして定義されていたが, 実際に `catkin build`/`catkin_make` されたことが一度もなく, `roslaunch` でパッケージ名を解決できるかは未検証だった. `/workspace/catkin_ws` を新規作成し (`src` を `/workspace/ros` へのシンボリックリンクとして作成, `.gitignore` に追加済みで追跡対象外), `catkin_make` を実行した. `rospack find bundlesdf_node`/`rospack find sam3_segmenter` が正しく解決できることを確認した. なお catkin_make が実ソースツリー内 (`ros/CMakeLists.txt`) に環境依存のシンボリックリンクを誤って作成していたことに気づき, これは削除した (コミット対象から除外済み).

### モックカメラストリームノードの新規実装

実カメラなしで `bundlesdf_node.py` を検証するため, `ros/bundlesdf_node/scripts/mock_camera_stream.py` を新規実装した. 既存のミルクデモ録画データ (`data/2022-11-18-15-10-24_milk/`, rgb/depth/masks 各1932枚 + `cam_K.txt`) を読み込み, `bundlesdf_node.launch` のデフォルト remap 先と同じトピック名 (`/d455_1/color/image_rect` 等) に, `decode_bgr`/`decode_depth`/`decode_mask` が期待するエンコーディング (`bgr8`/`16UC1`/`mono8`) で配信する. `ApproximateTimeSynchronizer` (slop=0.05) に噛み合うよう, 全フレームで同一 stamp を付与する設計にした.

### 実行時バグの発見と FTA による根本原因の確定

上記のモックストリームを使い, 実際に `bundlesdf_node` を起動して動作検証したところ, `~object_pose` トピックへの publish が100%決定的に失敗することが判明した (10フレーム×2回, 100フレーム×2回, 計172フレーム超で毎回再現).

Fault Tree Analysis で根本原因を確定した. `bundlesdf_node.py` の `on_frame()` (修正前の145-151行目) は `tracker.run()` 呼び出し直後に `os.path.exists(f'{debug_dir}/ob_in_cam/{id_str}.txt')` をチェックしていたが, 実際のファイル書き込みを行う `bundlesdf.py` 側の `saveNewframeResult()` は性能改善コミット `e894df9` (2026-07-03) でバックグラウンドスレッドへの非同期キューイングに変更されており, その完了保証 (`self._save_result_queue.join()`, `bundlesdf.py:607`) は次フレームの `process_new_frame()` 冒頭でしか行われない. `bundlesdf_node.py` は1フレーム1コールバックで完結する構造のため, この完了保証が成立する前に必ずファイル存在チェックを行ってしまう. `bundlesdf_node.py` (コミット `e243645`) は非同期化コミット (`e894df9`) の約4.5時間前に書かれており, 後から入った性能改善が前提としていたファイル完了保証を壊した形になる.

修正は, ファイルへの書き込み・存在チェック・読み込みという迂回を廃止し, `run()` が返った時点で同期的に更新済みの `self.tracker.bundler._newframe._pose_in_model` をメモリから直接読むよう変更した. GUI モード用のコードパス (`bundlesdf.py:864,868`) が既に同じパターン (`np.linalg.inv(frame._pose_in_model)` をメモリから直接読む) を使っていたことから着想を得た. 修正後, 20フレームのモックストリームで19/19フレームが実際に `PoseStamped` を publish し, `produced no pose` 警告は0件になったことをライブ検証で確認した. この修正は `ros/bundlesdf_node/scripts/bundlesdf_node.py` に対する変更であり, 既に作業ツリーに反映されている (コミットは別途行う).

### 副次的に発見した未解決の問題

上記の100フレーム検証中, 処理された74フレーム中40フレームで BundleTrack 内部のトラッキング喪失 (`_cloud_down points#: 0 too small ... mark as FAIL`) が発生していた. これは publish バグとは無関係な別問題であり, ミルクデモのモックデータ特有の追跡品質の問題である可能性がある. 原因は本セッションでは未調査のまま `notes/ISSUES.md` に新規追加した.

### 未検証事項

上記は全て録画データのモック配信による検証であり, 実際のカメラハードウェア (osx 側) からのライブストリームでの検証はまだ行っていない.
