# アーキテクチャと学習パイプライン（仕様書）

内部開発メモ（公開 mdbook には含まれない）。最終更新: **2026-07-15**。
他の AI エージェント / 開発者への引き継ぎを想定した仕様書。入口は [CLAUDE.md](../../CLAUDE.md)、改善計画と実験台帳は [roadmap.md](roadmap.md)。

## 1. ファイルマップ（mortal/）

| ファイル | 役割 |
|---|---|
| `train.py` | 学習ループ本体（オフライン/オンライン共通）。損失計算、checkpoint 保存、test_play / suite 起動、TensorBoard 出力。`main()` は子プロセス監視ループ（オンラインのハング対策で `sys.exit(0)` 再起動） |
| `model.py` | `Brain`（ResNet+ChannelAttention エンコーダ、v1–v4、oracle 入力対応）、`DQN`（dueling ヘッド）、`AuxNet`、`CategoricalPolicy`、`GRP`（GRU 順位予測器） |
| `dataloader.py` | `FileDatasetsIter` — mjai ログ(.json.gz) → 学習サンプル。GRP による局報酬の付与もここ。GRP 構築は `build_iter` 内（Windows DataLoader worker 対策、`__init__` に戻すと壊れる） |
| `reward_calculator.py` | GRP の出力から局単位の期待 pt 差分報酬を計算 |
| `engine.py` | `MortalEngine`（対局用推論エンジン: Boltzmann 探索、bucket batch、CUDA Graphs）、`build_engine_from_state`（checkpoint→エンジン復元。DQN/Policy 両ヘッド対応）、`resolve_amp_dtype` |
| `player.py` | `TestPlayer`（評価対局）、`TrainPlayer`（自己対戦データ生成、対戦相手プール）。cudnn.benchmark の復元は両者とも try/finally 済み |
| `client.py` / `server.py` / `common.py` | 分散オンライン学習（ワーカー / バッファサーバ / 通信プロトコル）。server は無認証なので信頼できる LAN でのみ bind_host を開くこと。submit_replay のファイル名は basename 化済み（パストラバーサル対策） |
| `suite.py` | **決定点ベンチマークスイート v1**。ホールドアウト牌譜の決定点で checkpoint を採点（§7 参照） |
| `suite_batch.py` | 複数 checkpoint の一括 suite 採点 → CSV/TB 出力（Phase A ステップ3、オフライン→オンライン遷移の分解用） |
| `train_baseline.py` | オラクル運ベースライン b(state, hidden info) の学習（**棚上げ済み機能**、実験記録は notes 参照。ウォームスタート転移のコードは補助タスクに再利用可能） |
| `train_grp.py` | GRP（報酬モデル）の学習 |
| `checkpoint.py` | `load_checkpoint()` — `weights_only=True` を維持しつつ numpy スカラー/dtype を allowlist（レガシー checkpoint 対応）。**torch.load を直接書かずこれを使う** |
| `one_vs_three.py` | 1対3 の評価対局スクリプト |
| `bench_inference.py` | 推論速度・行動一致率ベンチ |
| `strip_checkpoint.py` | checkpoint から optimizer 等を除去 |
| `lr_scheduler.py` | 線形ウォームアップ + コサイン減衰 |
| `config.py` | `MORTAL_CFG` 環境変数（デフォルト `config.toml`）から TOML を読む。全モジュールが `from config import config` で共有 |
| `start_all.ps1` / `start_worker.ps1` / `stop_worker.ps1` | 常設運用スクリプト（相対パス化済み）。start_all の再起動ループは無条件（設定ミス時に永久クラッシュループになる点に注意） |

libriichi（Rust）:

| モジュール | 役割 |
|---|---|
| `dataset` | `GameplayLoader` = mjai ログ→特徴量、suit 置換 augmentation。`emit_sp_labels=true`（デフォルト off）で各決定点に **37 次元の SP 打牌 EV ベクトル**（打牌行動空間 0–36 整列、非候補は NaN、`PlayerState::sp_discard_evs` 参照）を追加出力。`invisible.rs` は**全員の手牌・向聴・待ち・フリテン・山の実際の並びをパース済み**（oracle 入力・後知恵ラベルの供給源） |
| `state` | 観測エンコーダ。**v4 のチャネル定義は `state/obs_repr.rs`**（河・手出し/ツモ切り・リーチ宣言牌・SP テーブル等が既にエンコード済み。詳細は notes セクション2–3） |
| `algo/sp` | SPCalculator（受け入れ・聴牌率/和了率/EV テーブル）。SHANTEN_THRES=3 のため**4向聴以遠はプレーンが空白**。2.3 倍高速化済み（ビット同一検証付き、commit b27aa5e） |
| `arena` | OneVsThree（自己対戦・評価対局） |
| `stat` | 対局統計（和了率・放銃率など test_play/* の元データ） |
| `consts` | `obs_shape(version)`, `ACTION_SPACE=46`, `GRP_SIZE` |

## 2. 行動空間（インデックスは train.py / suite.py のメトリクスでも使用）

```
0–36: 打牌（34 通常 + 34/35/36 = 赤5m/5p/5s） / 37: リーチ / 38–40: チー / 41: ポン / 42: カン / 43: 和了 / 44: 流局 / 45: パス
```

## 3. オフライン学習のデータフロー

```
.json.gz (mjai) → GameplayLoader → Gameplay(per game)
  → FileDatasetsIter: GRP で局ごとの期待pt差分を報酬化 → サンプル
  → DataLoader → train_batch
```

- サンプル = `(obs, action, mask, steps_to_done, kyoku_reward, player_rank, at_turn, shanten)`。`FileDatasetsIter(include_sp_labels=True)`（suite の sp_metrics 用、学習ではオフ）のときのみ末尾に 37 次元 SP 打牌 EV ベクトルが付く。この kwarg が False のときは `GameplayLoader` に `emit_sp_labels` を渡さないため、旧 pyd でも学習は動く
- **同一局内の全着手は同じ kyoku_reward を受け取る**（局内のクレジット割り当てなし）
- `dataset.holdout_files`（1行1パスのテキスト）に列挙したファイルは**学習から除外**され suite 専用になる。train.py / train_baseline.py の両方で normcase/normpath 突き合わせで除外（パス不一致時は warning）
- **落とし穴**: buffer が file_batch_size=15（≈60局）単位のため、同一局のサンプル（ターゲット完全共有）が近接バッチに反復する。**回帰ヘッドの SGD 学習は丸暗記する**（C-1 実験で実証）。補助タスクを足すときは reserve_ratio > 0 / file_batch_size 増 / held-out 検証が必須

## 4. 損失（train.py `train_batch`）

- ターゲット: `q_target_mc = γ^steps_to_done × kyoku_reward`（純粋 MC、ブートストラップなし。γ は `[env] gamma`、通常 1）
- `dqn_loss`: デフォルト `0.5*MSE(Q(s,a), target)`。`[offline_loss] huber_delta > 0` で HuberLoss（二次領域 0.5e² で MSE とスケール一致）。**Huber はオフラインで有効と実証済み・存続**。オンライン持ち込みは要注意（和了報酬 = 正の外れ値を切る守備化バイアス、notes セクション4-3c）
- `cql_loss`（**オフラインのみ**、重み `[cql] min_q_weight`）: `logsumexp(Q) − Q(a)` = softmax(Q) の logged action への CE = BC 正則化
  - `[offline_loss] awbc = true` でサンプル毎重み `w = exp(clamp(adv/β))` 付き（**実験の結果、実質効果なしと決着**。roadmap の失敗台帳参照）。adv = MC リターン − 合法手平均 Q（detach、= dueling V と厳密一致）。`awbc_baseline = true` で adv のベースラインをオラクル b(s,z) に差し替え（**棚上げ**）
- `next_rank_loss`（重み `[aux] next_rank_weight`）: 局終了時順位の CE（AuxNet、現状唯一の補助タスク）
- オンラインでは CQL 項なし・trainee 席のデータのみ・`freeze_bn` を true にする運用

## 5. オンライン学習ループ

```
trainer (train.py online=true)          server.py              client.py ×N
  submit_param ──────────────────────→ param 保持 ←──────────── get_param
  drain() ←──── buffer→drain 移動 ←─── buffer_dir ←──────────── submit_replay
  drain したログで1パス学習 → submit_param → （繰り返し）
```

- ワーカーは trainee 1席 vs ベースライン3席（`opponent_pool_prob` で過去 checkpoint とも対戦可）。探索は ε-Boltzmann（`[train_play]`、現行 ε=0.005 / temp=0.05 ≈ 探索ほぼゼロ）
- `force_sequential = false` で生成と学習が同時進行（高速化）。`drain_min_count` で drain の最小ファイル数を制御
- `sample_reuse_rate / sample_reuse_threshold` は**キーのみ存在・未実装**
- 既知バグ: test_play 後にハング → `sys.exit(0)` で再起動（`main()` が子プロセスを張り直す。原因未特定）
- ワーカー停止はいつでも安全（未完セッションが消えるだけ）

## 6. Checkpoint 形式（state_file）

dict keys: `mortal`（Brain）, `current_dqn`, `aux_net`, `optimizer`, `scheduler`, `scaler`, `steps`, `timestamp`, `best_perf`, `top_checkpoints`, `config`（**config 全体が埋め込まれる**）。

- `build_engine_from_state` はこの `config` と state dict の内容（BN 統計の有無→norm 判定、`policy_net`/`current_dqn`→ヘッド判定）からエンジンを復元する。**後方互換を壊さないこと**
- `async_save = true` で背景スレッド保存（tmp 書き込み + atomic replace）。test_play で best 更新時は**更新後の best_perf で再キューされる**（2026-07 修正済み）
- top-k 候補は `best_state_file` の隣の `candidates/` に保存（`[control] top_k`）。resume 時に孤児 candidate_*.pth を掃除する（2026-07 修正済み）
- ロードは必ず `checkpoint.py load_checkpoint()`（weights_only=True + numpy allowlist）

## 7. 評価系（3層）

計測は「大きな差 → test_play、小さな差 → suite、実戦最終確認 → one_vs_three」の3層構造。

### test_play（対局評価・ノイズ大）
- `TestPlayer.test_play`: 固定シード（10000〜/key 0x2000）、1対3で4席ローテーション → **checkpoint 間でペア比較になっている**。`aggregate_runs` で複数バッチ
- train.py が順位分布から `avg_rank`/`avg_pt` の標準誤差を閉形式で計算し `test_play/avg_ranking_se` / `avg_pt_se` に出力。3000 試合で SE ≈ 0.02。`[test_play] gate_sigma > 0` で best/top-k 更新を「SE×gate_sigma を超えた改善」に制限（0=従来挙動）
- 注意: SE は iid 仮定（4席複製+固定シードの相関で厳密には過小/過大あり、目安として使う）。gate_sigma を上げすぎると pt と rank の**両方**のマージン超えが必要なため best が更新不能になり得る

### suite（決定点ベンチマーク・ノイズほぼゼロ）— 2026-07 新規
- `suite.py`: `dataset.holdout_files` の牌譜の全決定点で checkpoint の argmax を採点。フォワード1回・数分・同一局面比較なので**ステップ単位の曲線が引ける**（test_play では見えない牌効率・押し引きの小さな変化を可視化）
- `[suite] enabled = true` で test_every 境界ごとに自動実行、`suite/*` に出力。単発は `python suite.py --state <pth>`
- 主要メトリクス:
  - `suite/efficiency/shanten_regression_rate`（無スレット・門前時の向聴後退率）、`advance_miss_rate`（向聴進行可能時に進行打牌を外した率）
  - `suite/agreement/human_top1*`（人間との一致率、スレット文脈・シャンテン別内訳）
  - `suite/behavior/*`（鳴き見送り率・リーチ率・和了率・対リーチ押し引き — **スタイル許容帯の監視に使う**）
  - `suite/q/*`（top1-top2 マージン、logged action との Q ギャップ）
  - `suite/sp/*`（`[suite] sp_metrics = true` のとき。単発実行は `--sp` で強制可）: `model_match_rate(_no_threat)` = モデル argmax と SP 最大 EV 打牌の一致率、`model_ev_loss(_no_threat)` = 選んだ打牌の SP EV 損失（点）、`model_off_table_rate` = 向聴後退打牌を選んだ率、`human_match_rate` / `human_ev_loss` = 人間 logged action の同指標（アンカー）、`coverage` = SP ラベルの存在率（向聴 4 以遠・自リーチ後・海底間際は NaN）。ラベルは GameplayLoader の `emit_sp_labels`（喰い替え・リーチ宣言の合法打牌でフィルタ済み、赤5は別スロット→Python 側で deaka 集約）。合法手外に EV が立っていたら即エラー（validate_channels と同思想）
- v4 obs チャンネル定数（cand=874, keep=875, next=876, own_riichi=878, opp_fuuro=771, opp_riichi=857）は経験的検証に基づく。`validate_channels()` が毎回不変条件（mask との整合 ≥0.99、keep/next 排反、fill-style 行）を再検証し、obs エンコーダ変更時は**黙って壊れず即エラー**
- v1 は version 4 専用

### one_vs_three（大規模実戦評価）
- `one_vs_three.py` + `[1v3]`: challenger vs champion(+akochan)。最終確認用

## 8. TensorBoard 主要タグ

`loss/*`（dqn, cql, next_rank）、`q/* target/* td/*`（Q とターゲットの分布）、`train/*`（行動率・シャンテン別・脅威対応・鳴き判断など。v4 のみ `train/threat/*`）、`train/awbc/*`（AWBC 有効時）、`grad/* param/* grad_to_param/*`（補助タスク干渉監視は `grad/aux_norm` / `grad/dqn_norm`）、`test_play/*`、`suite/*`、`perf/*`（steps_per_sec, gpu_mem, save/test/suite 時間）。
`metrics_every` で詳細メトリクスの間引き可（1 = 従来挙動）。

## 9. ハードウェア・運用トポロジ

- 常設（旧 PC, 172.16.10.28 固定 IP）: RTX 3060 Ti 8GB / i5-3470 4C4T / 16GB。server + trainer + client 同居、`start_all.ps1` で自動再起動
- 増援（メイン PC）: 5600X + 5060 Ti。`start_worker.ps1` / `stop_worker.ps1` で随時参加・離脱。server は `bind_host='0.0.0.0'` で LAN 待受け
- **自己対戦は CPU 律速**（実測: 400 同時ゲームで推論 27%、残りは Rust 単一スレッドのゲームループ+Python 接合。旧 PC 1 client ≈ 6,300 games/h）
- 推論高速化の採用状況: autocast + bucket batch + CUDA Graphs = **採用**（小バッチで5〜6倍、行動一致率 1.0000）。`weights_dtype='bfloat16'` = **不採用**（行動一致率 0.9639 で基準未達）

## 10. 既知の未修正問題（優先度低、2026-07-10 レビューより）

- 条件付きメトリクス（train/threat/*, train/call/*）は条件が空のバッチでも分母に数えられ過小表示される
- `legal_action_count_max` は「バッチ内 max の平均」で名前と意味がずれている
- train.py に is_call_action 等を再計算する死にコードあり
- torch.quantile の約1677万要素上限により、save_every × batch_size が大きい設定で all_td_abs が RuntimeError になり得る
- `extract_threats`（train.py）のチャンネル 771/857 はハードコードで検証なし（suite.py の validate_channels と異なり、エンコーダ変更時に黙って誤る）
- Rust `shanten_deltas_plus`: tehai[tid]==4 で base-5 インデックスが桁あふれするが「壁に5枚目はない」呼び出し規約で成立（debug_assert 推奨）
- start_all.ps1 の無条件再起動ループ（設定ミスで5秒ごと永久クラッシュ）
