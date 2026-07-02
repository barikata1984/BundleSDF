# インターフェース契約: ros-one online 化 (feat/ros-one-online)

本ファイルは f1-container / f2-segmenter / f3-matcher の並列実装に先立ち,
コンポーネント間のインターフェースを凍結したものである.
凍結後に契約を変更する場合は interface hub 経由で再交渉すること.

## 確定要件 (ユーザー決定済み, 交渉対象外)

- メインコンテナ: Ubuntu 22.04 jammy + ROS One (ros.packages.techfak.net) + CUDA 12.9
  + torch 2.8.0+cu129 (cp310) + kaolin 0.18.0 公式 wheel
  + OpenCV 4.13.0 縮小 CUDA ビルド (`/opt/opencv-cuda` prefix, `CUDA_GENERATION` build-arg).
- pytorch3d は依存として入れない (3 関数をベンダリング).
- セグメンタは SAM 3, 別コンテナ (noble + ROS One noble + Python 3.12),
  transformers のストリーミング concept API 使用, cv_bridge 不使用.
- EfficientLoFTR 既定, 旧 LoFTR は比較用に残す. `bundlesdf.py` は無改変.
- コンテナビルド実行・重みダウンロードはスコープ外. ckpt/モデルはコミットしない.

---

## S1 メインコンテナの供給物

### S1-A pytorch3d transforms ベンダリング (f1-container が作成, `nerf_helpers.py` が使用)

- 配置: リポジトリ root 直下の Python パッケージ `pytorch3d_transforms/`.
  - `__init__.py` で `so3_log_map`, `so3_exp_map`, `se3_exp_map` を再エクスポート.
  - 実装補助モジュール (so3 / se3 / rotation_conversions / math 等) は f1-container 裁量で分割してよい.
- import 差し替え: `nerf_helpers.py:15` を次に変更する.
  ```python
  from pytorch3d_transforms import so3_log_map, so3_exp_map, se3_exp_map
  ```
- 3 関数すべてを再エクスポートする. `so3_log_map` / `so3_exp_map` は現状未使用
  (実使用は `nerf_helpers.py:150` の `se3_exp_map` のみ) だが, import 文互換のため残す.
- 全リポジトリで pytorch3d.transforms の消費者はこの import 1 箇所のみ (hub が grep 確認).
  他コンポーネントは pytorch3d.transforms を使わないため, import パス整合の追加調整は不要.

### S1-B pip 依存の一元管理 (f1-container 所有)

- pip パッケージは `docker/ros-one.dockerfile` で一元管理する.
- f3-matcher (EfficientLoFTR) 由来の追加依存:
  - **新規追加は `loguru` の 1 件のみ**.
  - 既存 dockerfile で充足済み (追加不要): `einops` (現 L263), `kornia` (現 L248).
    EfficientLoFTR の実行時 import は torch / einops / loguru / kornia.
  - `gdown` は重み取得手順 (README) 用途のみで実行時不要. 採否は f1-container 裁量.

### S1-C 重みファイルの配置規約 (イメージに焼かない)

- EfficientLoFTR: `BundleTrack/EfficientLoFTR/weights/eloftr_outdoor.ckpt`
- 旧 LoFTR: `BundleTrack/LoFTR/weights/outdoor_ds.ckpt` (現状不変)
- 規約:
  - `weights/` は `.gitkeep` で作成し, ckpt 自体はコミットしない・イメージにも焼かない.
  - 実行時にボリュームマウントで供給する. dockerfile で重みを COPY しない.
  - 入手手順は `BundleTrack/EfficientLoFTR/README.md` に記載 (gdown 例など).

---

## S2 マスクトピック契約 (f2-segmenter 所有)

将来の BundleSDF ノード (今回スコープ外) が購読する前提で自己完結に定義する.

### 購読 (入力)

- トピック: `~image_in` (既定 remap `/camera/color/image_raw`).
- 型: `sensor_msgs/Image`, encoding `rgb8` または `bgr8`.
- QoS/処理: `queue_size=1`, 最新フレームのみ処理.
- 変換: cv_bridge 不使用, `np.frombuffer` で直接変換.

### 配信 (出力) = マスク契約

- トピック: `~mask_out` (既定 remap `/sam3/mask`).
- 型: `sensor_msgs/Image`.
- encoding: **`mono8`**, 値は **0 / 255** (前景=255), `step=width`, 単一チャネル.
- **header: 入力 RGB の header を完全継承** (stamp・frame_id をそのままコピー).
  下流が RGB / depth / mask を stamp で時刻同期できるようにするための最重要項目.
- 単一オブジェクト前提: 複数検出時は `postprocess_outputs` の scores 最大の 1 個を採用.
  scores が同点/空なら面積最大にフォールバック.
- 無検出フレーム: 全ゼロ mask を header 継承で配信する (配信自体はスキップしない).
  下流の時刻同期を切らさないため. 下流は `>0` 二値化で前景空と解釈する.

### 初期化インターフェース

- テキストプロンプト: `~text_prompt` (string, 必須, rosparam). SAM3 concept prompt.
- bbox / point / mask による初期化は本契約では定義しない.
  理由: 指定の SAM3 concept streaming API (`Sam3VideoModel` / `Sam3VideoProcessor`) は
  プロンプト追加が `add_text_prompt` のみで, box/point プロンプト口を持たない
  (一次ソース: HF transformers docs で確認).
  box/point/mask は別モデル `Sam3TrackerVideo` (SAM2 系譜, Promptable Visual Segmentation) の
  `add_inputs_to_inference_session(input_boxes=..., input_points=...)` にのみ存在する.
- 将来方針 (実装しない, README にも記載): インスタンス精密指定 (bbox/point/mask) が必要になった場合は,
  concept モデルから `Sam3TrackerVideo` (PVS 系) への差し替えで実現する.

### 参考 (実装側, 契約に影響なし)

- `postprocess_outputs` の返り値 dict: object_ids (int64), scores (float32),
  boxes (float32, N×4 XYXY 絶対 px), masks (bool, N×H×W 元解像度), prompt_to_obj_ids.

---

## S3 マッチャー切り替え契約 (f3-matcher 所有)

### S3-A `LoftrRunner.predict` の入出力契約 (凍結, 既存維持)

- 入力: `rgbAs`, `rgbBs` = `(N, H, W, C)` の `np.ndarray` (uint8 RGB).
- 出力: 長さ N のリスト. 各要素は `(M, 5)` float32, 列は `[x0, y0, x1, y1, conf]`.
- 消費側: `bundlesdf.py` の `find_corres` が `corres[i_pair][:,:4]` を消費 (`bundlesdf.py` 無改変).
- EfficientLoFTR の /32 パディング後に座標を元解像度系へ戻す責務は wrapper 内で閉じる.
- 上記契約は eloftr / loftr 両バックエンドで完全に維持する.

### S3-B バックエンド切り替え機構

- `loftr_wrapper.py` に `LoftrRunner(backend: str | None = None)` を追加する.
- `backend=None` のとき環境変数 `BUNDLESDF_MATCHER` を参照する
  (既定 `'eloftr'`, 値は `'eloftr'` | `'loftr'`).
- `bundlesdf.py` は `LoftrRunner()` 無引数呼び出しのまま無改変. 既定で eloftr が有効.

### S3-C ベンダリングと依存

- ベンダリング先: `BundleTrack/EfficientLoFTR/` (Apache-2.0 LICENSE 同梱).
- 訓練専用コード (`src/loftr/utils/supervision.py` 等) は除外し, 相対 import を閉じさせる.
  `detect_NaN` は最小スタブで解決. 以上は実装側の処理で契約に影響なし.
- pip 依存: S1-B のとおり `loguru` の追加のみ (einops / kornia は充足済み).
- 重みパス: S1-C のとおり `BundleTrack/EfficientLoFTR/weights/eloftr_outdoor.ckpt`.

---

## 凍結状態

- S1 (S1-A / S1-B / S1-C): f1-container 合意済み.
- S2: f2-segmenter 合意済み. bbox 初期化は team-lead 裁定により契約から除外 ((A) 採用).
- S3 (S3-A / S3-B / S3-C): f3-matcher 合意済み.

3 シームすべて収束. 本契約を凍結する.
