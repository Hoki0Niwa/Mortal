# Valueベース・オンライン深層探索

main の単一 DQN checkpoint を初期値にし、Actor-Critic を導入せずに、模倣データの
支持外へ探索を広げるオンライン実験。旧形式と旧挙動は config デフォルトで維持する。

## 採用した構成

1. **Bootstrapped DQN**: 共有 Brain に複数の dueling Q head を接続する。各サンプルは
   Bernoulli mask で一部の head のみを学習する。
2. **Randomized prior**: head ごとにパラメータを固定したランダム DQN を加算する。
   `Q_k = Q_learned,k + prior_scale * Q_prior,k`。prior のパラメータと、その出力から
   Brain への勾配は止める。軽量化のため入力は共有 Brain の表現を使うので、入力空間
   全体で固定した別 backbone の prior ではなく latent-space prior である。
3. **半荘単位の探索**: 自己対戦時は半荘開始時に head を一つ選び、その半荘の全着手で
   同じ head を使う。評価・通常推論は head 平均を使う。
4. **半荘 MC return**: 従来の当該局だけの報酬ではなく、現在局から終局までの GRP
   順位効用差を合計する。これは「終局時順位効用 − 現在局開始時の期待順位効用」に
   telescope するため、既存の報酬尺度を保ちながら局をまたぐ選択を学習できる。
5. **Population self-play**: `opponent_pool_dir = 'auto'` で `best.pth` 隣の
   `candidates/` を使う。初期 main モデルもこのディレクトリへ固定コピーしておく。

## 推奨する初回設定

`config.example.toml` を別PCの実験ディレクトリへコピーし、少なくとも次を変更する。

```toml
[control]
online = true
state_file = 'D:/MortalIO/online-main/run/mortal.pth'
best_state_file = 'D:/MortalIO/online-main/run/best.pth'
tensorboard_dir = 'D:/MortalIO/online-main/logs/tensorboard'

[train_play.default]
boltzmann_epsilon = 0.0
opponent_pool_dir = 'auto'
opponent_pool_prob = 0.5

[online_rl]
bootstrapped_dqn = true
num_heads = 5
prior_scale = 0.1
bootstrap_prob = 0.8
sample_head_per_game = true
reward_mode = 'hanchan_return'

[test_play]
gate_sigma = 1.5
log_dir = 'D:/MortalIO/online-main/logs/test-play'

[online.remote]
host = '127.0.0.1'
port = 5000

[online.server]
bind_host = '127.0.0.1'
buffer_dir = 'D:/MortalIO/online-main/replay/buffer'
drain_dir = 'D:/MortalIO/online-main/replay/drain'
capacity = 800
force_sequential = true
drain_min_count = 800
reset_buffer_on_start = false
```

`train_play.default.log_dir`、baseline、GRP のパスも別PCの実在パスへ変更する。
初回は一台のPCで server / trainer / client を動かす。追加 worker を LAN から接続する
場合だけ `bind_host = '0.0.0.0'` とし、worker 側の `online.remote.host` を trainer PC の
LAN IP にする。通信には認証も暗号化もないため、インターネットへ直接公開しない。

## checkpoint の準備

元モデルを直接 `state_file` にすると上書きされる。次のコマンドで step、best 記録、
top-k をリセットした作業用 checkpoint と SHA-256 manifest を作る。

```powershell
Set-Location C:\path\to\Mortal\mortal
python prepare_online_seed.py `
  D:\models\candidate_950000_offline.pth `
  D:\MortalIO\online-main\run\mortal.pth
New-Item -ItemType Directory -Force `
  D:\MortalIO\online-main\run\candidates | Out-Null
Copy-Item D:\MortalIO\online-main\run\mortal.pth `
  D:\MortalIO\online-main\run\candidates\initial_main.pth
```

初回ロード時、旧単一 DQN の重みは全 trainable head へ複製される。prior は trainer が
一度だけ生成し、直後に server 経由で worker へ同じ state dict が配布される。以後の
checkpoint には trainable head と prior head の両方が保存される。

## 起動

`start_all.ps1` は絶対パスを埋め込まず、`$PSScriptRoot` と引数を使う。

```powershell
powershell -ExecutionPolicy Bypass -File .\start_all.ps1 `
  -PythonExe C:\path\to\envs\mortal\python.exe `
  -ConfigPath D:\MortalIO\online-main\config.toml
```

server 再起動時のリプレイ消失を避けるには `reset_buffer_on_start = false` を使う。
追加 worker も population を使う場合、`candidates/` は各 worker の同一設定パスへ同期
するか共有ストレージに置く。checkpoint 自体は現行プロトコルでは配信しない。

## 採否ゲート

- 評価方策は常に head 平均で固定する。
- 初期 main checkpoint と固定 seed の 1対3評価を比較する。
- `avg_rank` と `avg_pt` の双方が `gate_sigma × SE` を超えて改善した checkpoint のみ残す。
- `train/ensemble/selected_q_std` がほぼ 0 なら head が崩壊している。
- `train/ensemble/greedy_disagreement_rate` が高止まりし、評価が悪化する場合は
  `prior_scale` を下げる。両指標が早期に 0 へ落ちる場合は上げる。

最初の A/B ではこの構成を一体として main と比較する。その後の切り分けは同一 seed
から `reward_mode`、prior、population の順に一要素ずつ無効化する。
