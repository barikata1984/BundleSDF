# BundleSDF アーキテクチャ分析：ポリシー統合のための知見

**日付:** 2026-01-28  
**目的:** BundleSDFの出力をACT/MLPポリシーの入力として利用する際の設計指針

---

## 1. 特徴点マッチングの仕組み

### 1.1 データフロー

```
EfficientLoFTR (Python)
  └─ mkpts0_f, mkpts1_f: (M, 2)  ← 2Dマッチ点
  └─ mconf: (M,)                  ← 信頼度
  └─ m_bids: (M,)                 ← バッチID

       ↓ _raw_matches として保存

Python → C++ 受け渡し
  └─ _raw_matches: (M, 4) uint16  ← [uA, vA, uB, vB]

       ↓ rawMatchesToCorres() で3D変換

C++ 内部 (_matches)
  └─ vector<Correspondence>       ← 2D + 3D座標
```

### 1.2 Correspondence構造体

```cpp
class Correspondence {
    float _uA, _vA, _uB, _vB;          // 2D座標
    pcl::PointXYZRGBNormal _ptA_cam;   // 3D点（カメラ座標系）
    pcl::PointXYZRGBNormal _ptB_cam;
    float _confidence;
    bool _isinlier;
};
```

**注意:** `_ptA_cam`, `_ptB_cam` はpybindで公開されていないため、Pythonから直接アクセス不可。

### 1.3 各段階での特徴点数

| 段階 | 典型的な数 | 備考 |
|------|-----------|------|
| EfficientLoFTR出力 | 数百〜数千 | 画像解像度・テクスチャ依存 |
| 3D変換後 | 数十〜数百 | 有効なdepthがある点のみ |
| RANSAC後 | 5〜30 | 幾何的整合性でフィルタリング |

---

## 2. RANSAC後の特徴点

### 2.1 「最低5点」の意味

**5点は「保証」ではなく「閾値」である。**

```cpp
// FeatureManager.cpp:1695-1699
if (matches_cur_pair.size() < min_match_after_ransac) {
    matches_cur_pair.clear();  // 5点未満なら全破棄（0点になる）
}
```

| inlier数 | 結果 |
|---------|------|
| ≥ 5 | 採用（その点数がそのまま残る） |
| < 5 | 全破棄（0点） |

### 2.2 RANSAC inlierの選定基準

| 条件 | パラメータ | 典型値 |
|------|-----------|--------|
| 3D距離 | `inlier_dist` | 5mm |
| 法線角度 | `inlier_normal_angle` | 30° |
| 変換量（隣接） | `max_trans_neighbor` | 20mm |
| 回転量（隣接） | `max_rot_deg_neighbor` | 30° |

### 2.3 `_matches` へのアクセス

RANSAC後、`_matches` は inlier のみで上書きされる。特別なフィルタリングなしでRANSAC後の点が取得可能。

---

## 3. NeRFと姿勢推定の関係

### 3.1 アーキテクチャ：並列構造

```
カメラ入力 ──► 特徴点マッチング ──► RANSAC ──► Bundle Adjustment ──► 姿勢出力
(RGB+Depth)         │                              ▲
                    │                              │
                    ▼                              │
                  NeRF学習 ◄────────────────────────┘
                    │              (姿勢を使って学習)
                    ▼
                 メッシュ出力
```

**重要:** NeRFは姿勢推定に必須ではない。特徴点ベースの姿勢推定はNeRFを一切参照しない。

### 3.2 姿勢推定の2段階

| 段階 | ソース | NeRF依存 |
|------|--------|---------|
| 1段階目 | 特徴点 + RANSAC + BA | **なし** |
| 2段階目 | NeRF photometric refinement | あり |

### 3.3 ROSトピックで出力される姿勢

```python
# bundlesdf_node.py:417
pose_matrix = np.array(latest_frame._pose_in_model)
```

- NeRF未完了時：特徴点ベースの姿勢
- NeRF完了後：**NeRF refinement済みの姿勢で上書き**

---

## 4. NeRF Refinementのリスク

### 4.1 崩壊時の動作

```python
# bundlesdf.py:1353-1354
# 閾値チェックは「rematch判定」のみ。姿勢は無条件で上書き。
self.bundler._keyframes[i_f]._pose_in_model = self.p_dict["optimized_cvcam_in_obs"][i_f]
```

**NeRFが崩壊した場合、崩壊した姿勢がそのまま適用される。** 閾値による拒否は行われない。

### 4.2 NeRF崩壊の兆候

- 推論時間の増加に伴うメッシュ形状の劣化
- 表面の穴、不自然な突起
- テクスチャの破綻

---

## 5. ポリシー入力としてのデータソース比較

### 5.1 形状情報

| データソース | NeRF依存 | 安定性 | 形状表現力 | 固定サイズ |
|-------------|---------|--------|-----------|-----------|
| NeRFメッシュ | ✓ | × 崩壊リスク | ◎ | × |
| **マスク内Depth点群** | **×** | **◎** | **○** | **✓** |
| RANSAC特徴点 | × | ◎ | △ 部分的 | × (要パディング) |

### 5.2 推奨：マスク内Depth点群

```python
def get_observed_object_points(depth, mask, K, n_samples=512):
    """NeRFを経由せず、観測データから直接物体点群を取得"""
    valid = (mask > 0) & (depth > 0.1) & (depth < 2.0)
    vs, us = np.where(valid)
    
    if len(vs) == 0:
        return np.zeros((n_samples, 3), dtype=np.float32)
    
    # ランダムサンプリング
    indices = np.random.choice(len(vs), min(n_samples, len(vs)), replace=False)
    zs = depth[vs[indices], us[indices]]
    
    # 2D → 3D
    xs = (us[indices] - K[0, 2]) * zs / K[0, 0]
    ys = (vs[indices] - K[1, 2]) * zs / K[1, 1]
    
    points = np.stack([xs, ys, zs], axis=1)
    
    # パディング
    if len(points) < n_samples:
        padding = np.zeros((n_samples - len(points), 3))
        points = np.vstack([points, padding])
    
    return points.astype(np.float32)
```

### 5.3 NeRFの用途別評価

| 用途 | NeRFの価値 |
|------|-----------|
| **オフライン処理**（データ収集後） | ◎ 高品質メッシュ、再実行可能 |
| **オンライン推論**（リアルタイム） | × 崩壊リスク、代替手段あり |

---

## 6. 推奨アーキテクチャ

### 6.1 ポリシー入力の構成

```python
observation = {
    'robot_state': joint_angles,                    # (14,)
    'object_pose': pose_6dof,                       # (7,) position + quaternion
    'object_points': sampled_depth_points.flatten(), # (1536,) = 512 × 3
}
```

### 6.2 NeRFの位置づけ

- **姿勢refinement**: 無効化推奨（崩壊リスク排除）
- **メッシュ出力**: デバッグ・可視化用のみ
- **ポリシー入力**: 依存しない

### 6.3 信頼できる出力

| 出力 | 信頼性 | 用途 |
|------|--------|------|
| 特徴点ベース姿勢 | ◎ | ポリシー入力 |
| Depth + マスク点群 | ◎ | ポリシー入力 |
| セグメンテーションマスク | ◎ | 点群生成、可視化 |
| NeRFメッシュ | △ | 可視化のみ |
| NeRF refined姿勢 | △ | 使用非推奨 |

---

## 7. 実装上の注意点

### 7.1 ROSトピック設計

| トピック | メッセージ型 | 更新頻度 |
|---------|-------------|---------|
| `/bundlesdf/object_pose` | `geometry_msgs/PoseStamped` | 30Hz |
| `/bundlesdf/object_points` | `sensor_msgs/PointCloud2` | 30Hz |
| `/bundlesdf/mask` | `sensor_msgs/Image` | 30Hz |

### 7.2 可変長入力の処理

RANSAC特徴点を使う場合：
- パディング + マスク方式
- または統計的集約（mean, std, min, max）

Depth点群を使う場合：
- 固定数サンプリング（例：512点）でそのまま使用可能

---

## 8. まとめ

1. **BundleSDFの姿勢推定はNeRFに依存しない**（特徴点ベースで独立動作）
2. **NeRF崩壊時、姿勢が汚染されるリスクがある**（無条件上書き）
3. **オンライン推論ではNeRF依存を避けるべき**
4. **Depth + マスク点群が最も堅実なポリシー入力**
5. **NeRFはオフライン用途・可視化に限定して使用**
