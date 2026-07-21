# アルゴリズム改善ロードマップ

2026-06-12 の調査で策定。原則: **計測ゲート（test_play SE）を通った変更だけを main に残す**。新機能は config でデフォルトオフ。

## Phase 0: 評価の信頼性 — ✅ 完了（main にマージ済み、commit 74f150d）

- test_play の順位分布から `avg_rank`/`avg_pt` の標準誤差を閉形式で計算し、TensorBoard（`test_play/avg_ranking_se`, `avg_pt_se`）とログに出力
- `[test_play] gate_sigma`（デフォルト 0 = 従来挙動）: best/top-k 更新を「記録を gate_sigma×SE 以上上回ったとき」に制限
- 運用: まず数回の評価で SE を観察してから `gate_sigma = 1.5〜2.0` を設定する。3000 試合で SE ≈ 0.02

## Phase 1: オフライン損失の改善 — ✅ 実装済み（`exp/offline-awbc` ブランチ）、A/B 検証待ち

- **Huber 損失**: `[offline_loss] huber_delta`（0 = 従来の 0.5*MSE）。MC ターゲットの巨大な分散・外れ値（大物手/放銃）対策
- **AWBC**: `[offline_loss] awbc` — CQL/BC 項を advantage 重み付き（`w = exp(clamp(adv/β))`、正規化あり）にして「良い手ほど強く模倣」。ベースラインはモデル変更なしで取れる「合法手平均 Q（= dueling V、detach）」
- 推奨初期値: `huber_delta = 1.0`、`awbc_beta = 1.0`、`awbc_weight_clip = 8.0`
- 切り分け順: Huber のみ → 効けば +AWBC（1変数ずつ）
- 監視: `train/awbc/ess` が 0.3 を切ったら `awbc_beta` を上げる
- デフォルト設定での旧実装との数値等価性は実装時に検証済み（allclose）

### A/B 実験プロトコル（再学習不要の方式）

1. 既存 run が control。実験側は現在の checkpoint をコピーして「継続学習 N step」を比較する
2. 実験用 config では **state_file / best_state_file / tensorboard_dir / test_play.log_dir をすべて別パス**にする（test_play.log_dir は評価毎に rmtree されるため共有厳禁）
3. 同一 GPU なら順番に実行で可（比較は step 基準）
4. 採否は `test_play/avg_ranking` の差が SE を超えるかで判定

## Phase 2: オンライン学習の安定化 — 未着手

優先順:

1. **オンラインでの弱い BC/CQL 正則化**（数行）: 現状オンラインは正則化ゼロの MSE のみで、小データへの過適合→方策急変→発振のリスク。`min_q_weight` のオンライン版キーを追加（デフォルト 0 = 従来）
2. **top-k candidates → opponent pool 接続**: コード変更不要。`[train_play] opponent_pool_dir` を candidates ディレクトリに向けるだけ（`_load_pool_opponent` は学習 checkpoint をそのまま読める）。config 文書化のみ
3. **サンプル再利用の復活**: `[online.server] sample_reuse_rate / sample_reuse_threshold` はキーだけ存在し server.py に実装が無い（フォークの改修で消失）。drain 時に一部ファイルをバッファへ残す upstream 相当の実装を入れる

## Phase 3: 大規模改修 — Phase 1–2 の効果確認後に判断

- **分位点ヘッド（分布型 RL / QR-DQN 風）**: DQN ヘッドの置換。麻雀の多峰的リターンに対し Huber より本格的な対策
- **greedy フラグのログ付与 + 探索手のマスク**: 現状 ε-Boltzmann の探索手も等しく学習されており、探索を強められない根本原因。libriichi（mjai ログ拡張 or サイドチャネル）と dataloader の改修が必要
- **TD(λ)/n-step + ターゲットネットワーク**: 局内のクレジット割り当て。オンラインで特に有効
- **GRP の自己対戦データでの再学習**: 自己対戦が進むと報酬モデルが OOD になる。train_grp.py は既存

## 検討済み・保留の案（再調査不要）

- **Oracle 蒸留**: `Brain(is_oracle=True)` と `take_invisible_obs()` のインフラは存在するが train.py 未使用。Suphx 流の蒸留は有望だが工事が大きい → Phase 3 以降
- **補助タスク追加**（放銃予測・相手テンパイ予測など）: `AuxNet` は dims タプル対応で追加は安価。Phase 1 の結果次第で挟む
- **checkpoint 平均（model soup）/ EMA**: top-k candidates の重み平均はほぼ無料で試せる。優先度中
- **行動種別の損失バランシング**: リーチ/鳴き/和了判断は打牌に比べ希少。oversampling は効く可能性があるが計測基盤が先
- **scheduler.step() がマイクロバッチ毎**（`opt_step_every > 1` のとき LR がバッチ数で消化される）: 意図確認が必要な軽微事項
- **エポック末尾の batch_size 未満の端数切り捨て**: 影響軽微、対応不要と判断

## 失敗実験の記録

- [SP 改造から SP3 までの実験・失敗記録](2026-07-21-sp-modification-through-sp3-failure-record.md) — SPCalculator 高速化、フルラベル評価、SPv2 補助教師、SP3 の速度・品質検証と中止理由。SPv2 / SP3 は廃止済み。
