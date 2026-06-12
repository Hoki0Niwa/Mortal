# アーキテクチャと学習パイプライン

内部開発メモ（公開 mdbook には含まれない）。最終更新: 2026-06-12

## ファイルマップ（mortal/）

| ファイル | 役割 |
|---|---|
| `train.py` | 学習ループ本体（オフライン/オンライン共通）。損失計算、checkpoint 保存、test_play 起動、TensorBoard 出力 |
| `model.py` | `Brain`（ResNet+ChannelAttention エンコーダ）、`DQN`（dueling ヘッド）、`AuxNet`、`CategoricalPolicy`、`GRP`（GRU 順位予測器） |
| `dataloader.py` | `FileDatasetsIter` — mjai ログ(.json.gz) → 学習サンプル。GRP による局報酬の付与もここ |
| `reward_calculator.py` | GRP の出力から局単位の期待pt差分報酬を計算 |
| `engine.py` | `MortalEngine`（対局用推論エンジン: Boltzmann 探索、bucket batch、CUDA Graphs）、`build_engine_from_state`（checkpoint→エンジン復元。DQN/Policy 両ヘッド対応） |
| `player.py` | `TestPlayer`（評価対局）、`TrainPlayer`（自己対戦データ生成、対戦相手プール） |
| `client.py` / `server.py` / `common.py` | 分散オンライン学習（ワーカー / バッファサーバ / 通信プロトコル） |
| `train_grp.py` | GRP（報酬モデル）の学習 |
| `one_vs_three.py` | 1対3 の評価対局スクリプト |
| `lr_scheduler.py` | 線形ウォームアップ + コサイン減衰 |
| `config.py` | `MORTAL_CFG` 環境変数（デフォルト `config.toml`）から TOML を読む。全モジュールが `from config import config` で共有 |

libriichi（Rust）: `dataset`（GameplayLoader = ログ→特徴量、suit 置換 augmentation）、`state`（観測エンコーダ、v4 のチャネル定義は `state/obs_repr.rs`）、`arena`（OneVsThree）、`stat`（対局統計）、`consts`（obs_shape, ACTION_SPACE=46）。

## 行動空間（インデックスは train.py のメトリクスでも使用）

0–36: 打牌 / 37: リーチ / 38–40: チー / 41: ポン / 42: カン / 43: 和了 / 44: 流局 / 45: パス

## オフライン学習のデータフロー

```
.json.gz (mjai) → GameplayLoader → Gameplay(per game)
  → FileDatasetsIter: GRP で局ごとの期待pt差分を報酬化 → サンプル
  → DataLoader → train_batch
```

サンプル = (obs, action, mask, steps_to_done, kyoku_reward, player_rank, at_turn, shanten)。
**同一局内の全着手は同じ kyoku_reward を受け取る**（局内のクレジット割り当てなし）。

## 損失（train.py `train_batch`）

- ターゲット: `q_target_mc = γ^steps_to_done × kyoku_reward`（純粋 MC、ブートストラップなし。γ は `[env] gamma`、通常 1）
- `dqn_loss`: デフォルト `0.5*MSE(Q(s,a), target)`。`[offline_loss] huber_delta > 0` で HuberLoss（二次領域 0.5e² で MSE とスケール一致）
- `cql_loss`（オフラインのみ、重み `[cql] min_q_weight`）: `logsumexp(Q) − Q(a)` = softmax(Q) の logged action への CE = BC 正則化
  - `[offline_loss] awbc = true` でサンプル毎重み `w = exp(clamp(adv/β))` 付き。adv = MC リターン − 合法手平均Q（detach、= dueling V と厳密一致）。log 空間 clamp + バッチ内 mean=1 正規化
- `next_rank_loss`（重み `[aux] next_rank_weight`）: 局終了時順位の CE

## オンライン学習ループ

```
trainer (train.py online=true)          server.py              client.py ×N
  submit_param ──────────────────────→ param 保持 ←──────────── get_param
  drain() ←──── buffer→drain 移動 ←─── buffer_dir ←──────────── submit_replay
  drain したログで1パス学習 → submit_param → （繰り返し）
```

- ワーカーは trainee 1席 vs ベースライン3席（`opponent_pool_prob` で過去 checkpoint とも対戦）。探索は ε-Boltzmann（`[train_play]`）
- オンラインでは **CQL 項なし**、trainee 席のデータのみ学習。`force_sequential` でほぼ on-policy 化
- 既知バグ: test_play 後にハング → `sys.exit(0)` で再起動（train.py 末尾の `main()` が子プロセスを張り直す）

## Checkpoint 形式（state_file）

dict keys: `mortal`（Brain）, `current_dqn`, `aux_net`, `optimizer`, `scheduler`, `scaler`, `steps`, `timestamp`, `best_perf`, `top_checkpoints`, `config`（**config 全体が埋め込まれる**）。
`build_engine_from_state` はこの `config` と state dict の内容（BN統計の有無→norm 判定、`policy_net`/`current_dqn`→ヘッド判定）からエンジンを復元する。**後方互換を壊さないこと。**

## 評価系

- `TestPlayer.test_play`: 固定シード（10000〜）、1対3で4席ローテーション → checkpoint 間でペア比較になっている。`aggregate_runs` で複数バッチ
- train.py が順位分布から `avg_rank`/`avg_pt` の標準誤差を閉形式で計算し `test_play/avg_ranking_se` / `avg_pt_se` に出力。`[test_play] gate_sigma > 0` で best/top-k 更新を「SE×gate_sigma を超えた改善」に制限（0=従来挙動）
- top-k 候補は `best_state_file` の隣の `candidates/` に保存（`[control] top_k`）

## TensorBoard 主要タグ

`loss/*`（dqn, cql, next_rank）、`q/* target/* td/*`（Q とターゲットの分布）、`train/*`（行動率・シャンテン別・脅威対応・鳴き判断など大量の挙動メトリクス。v4 のみ `train/threat/*`）、`train/awbc/*`（AWBC 有効時: adv_mean/adv_std/weight_std/clip_rate/ess）、`grad/* param/* grad_to_param/*`、`test_play/*`、`perf/*`（steps_per_sec, gpu_mem, save/test 時間）
