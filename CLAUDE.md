# CLAUDE.md

このリポジトリは [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal)（深層強化学習による日本麻雀AI）の**個人フォーク**。
上流との差分は主に**学習環境の拡張**: TensorBoard メトリクス大幅増、分散自己対戦ワーカー、top-k チェックポイント管理、推論高速化（CUDA Graphs / bucket batch）、評価の統計的ゲート（gate_sigma）、オフライン損失オプション（Huber / AWBC）、決定点ベンチマークスイート（suite.py）。

## 最初に読むドキュメント（引き継ぎ必読・この順で）

1. [docs/dev/architecture.md](docs/dev/architecture.md) — コード構成・データフロー・損失の数式・checkpoint 形式・評価系の**仕様書**
2. [docs/dev/roadmap.md](docs/dev/roadmap.md) — **実験台帳（成功/失敗の記録と理由）**と性能改善ロードマップ。**提案の前に必ず失敗台帳を確認すること**
3. [docs/dev/2026-07-10-review-and-training-notes.md](docs/dev/2026-07-10-review-and-training-notes.md) — コードレビュー結果と学習戦略の詳細分析（roadmap の根拠資料）

## 構成

- `libriichi/` — Rust コア（ゲームエンジン、mjai ログパーサ、特徴量エンコーダ obs_repr.rs、シャンテン/SP 計算、アリーナ、統計）。Python から `libriichi` モジュールとして import される
- `mortal/` — Python の学習・推論コード一式（ファイル別の役割は architecture.md の表）
- `docs/src/` — **公開 mdbook**。内部メモを置かない（内部メモは `docs/dev/`）

## 環境とコマンド

- Windows 10 / PowerShell。conda env `mortal`（`C:\Users\hoki0\miniconda3\envs\mortal\python`、Python 3.12 + torch + toml/tqdm/tensorboard）。システム Python は使わない
- ハード: RTX 3060 Ti 8GB + i5-3470（4C4T）+ RAM 16GB。自己対戦は **CPU 律速**。メイン PC（5600X + 5060 Ti）を LAN ワーカーとして随時追加
- libriichi ビルド: `cargo build -p libriichi --lib --release` → `cp target/release/riichi.dll mortal/libriichi.pyd`。conda env `mortal` 以外の Python が PATH 先頭の環境では **`PYO3_PYTHON=C:\Users\hoki0\miniconda3\envs\mortal\python.exe` を付ける**（pyo3 が別バージョンの python DLL にリンクし import 時に「指定されたモジュールが見つかりません」で落ちる）。学習稼働中は pyd がロックされるため差し替えは停止後に
- Rust テスト: `cargo test --workspace --no-default-features --features flate2/zlib`
- 学習系の実行はすべて `mortal/` をカレントにして行う（`libriichi.pyd` と `config.toml` がそこにあるため）
- config は env `MORTAL_CFG`（デフォルト `./config.toml`）で指定。**実 config はコミットしない**。リポジトリにあるのは `mortal/config.example.toml` のみ
- オフライン学習: `python train.py`（`[control] online = false`）
- オンライン学習: `python server.py`（バッファ）+ `python client.py`（自己対戦ワーカー、複数可・LAN 可）+ `python train.py`（`online = true`）
- 決定点ベンチマーク単発実行: `python suite.py --state /path/to/checkpoint.pth`
- Python 構文チェック: `python -c "import ast; ast.parse(open(r'mortal\train.py', encoding='utf-8').read())"`

## 鉄則

1. **main は稼働中の学習環境**。アルゴリズム実験は `exp/*` ブランチで行い、評価ゲートを通ったものだけ main にマージする
2. **新しい config キーは必ず `.get()` + デフォルト値**で読み、デフォルトは旧挙動を bit 同一に再現する。新機能はデフォルトオフ。理由: checkpoint に config dict が埋め込まれ、`build_engine_from_state`（mortal/engine.py）が古い checkpoint の config を読むため
3. **checkpoint（state_file）のフォーマットを変えない**。やむを得ず変える場合は `build_engine_from_state` と train.py の resume 処理の両方を後方互換に保つ
4. **新しい config キーは `mortal/config.example.toml` にコメント付きで文書化する**（このフォークの慣習）
5. **学習改善の効果判定は計測ゲート基準**: 大きな差は test_play の SE（3000 試合で SE ≈ 0.02、TensorBoard `test_play/avg_ranking_se`）、小さな差（牌効率・挙動）は suite/*（ノイズほぼゼロの同一局面比較）。SE 未満の test_play 差はノイズとして扱う
6. **失敗済み施策を再提案しない**（roadmap.md の失敗台帳参照）。再挑戦するなら「前回なぜ失敗したか」と「今回何が違うか」の説明とセットで
7. `torch.load` は常に `weights_only=True`（レガシー checkpoint は `checkpoint.py` の `load_checkpoint()` を使う）。opponent pool ディレクトリは .pth を置くだけで読み込まれるため、`weights_only=False` は任意コード実行の入口になる

## 既知の問題・落とし穴

- train.py: オンラインモードで test_play 後にプロセスがハングする既知バグ → `sys.exit(0)` で子プロセスごと再起動する設計（`main()` が監視ループ）。原因未特定
- dataloader.py: GRP の構築を `__init__` でなく `build_iter` で行うのは Windows の DataLoader worker 対策。`__init__` に戻すと壊れる
- `test_play.log_dir` は評価のたびに `rmtree` される。**トレーナー2本でこのディレクトリを共有してはいけない**（A/B 実験時は state_file / best_state_file / tensorboard_dir / test_play.log_dir をすべて分離）
- **回帰系の補助タスクはこのデータパイプラインで丸暗記する**（file_batch_size=15 ≈ 60局単位のバッファで同一局のターゲットが近接バッチに反復するため）。held-out ファイルでの検証を必須とし、reserve_ratio / file_batch_size / 凍結トランクで対策（roadmap.md 参照）
- `[online.server] sample_reuse_rate / sample_reuse_threshold` は**未実装**（キーだけ存在。server.py に実装なし）
- `player_names_files` / ホールドアウトリスト等のテキストは `utf-8-sig` で読む（BOM 対策。デフォルトの cp932 は壊れる）
- DQN は dueling 構造で、`a_mean` を合法手上で取るため「合法手の Q 平均 ≡ V ヘッド出力」が厳密に成り立つ
- suite.py の v4 obs チャンネル定数（874-878, 771, 857）は経験的に検証したもの。`validate_channels()` が毎回不変条件を再検証し、エンコーダ変更時は即エラーで落ちる設計。train.py の `extract_threats`（771/857）には同等の検証がないので注意
