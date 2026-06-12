# CLAUDE.md

このリポジトリは [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal)（深層強化学習による日本麻雀AI）の個人フォーク。
上流との差分は主に**学習環境の拡張**: TensorBoard メトリクス大幅増、分散自己対戦ワーカー、top-k チェックポイント管理、推論高速化（CUDA Graphs / bucket batch）、評価の統計的ゲート、オフライン損失の改善オプション。

詳細ドキュメント（先に読むこと）:
- [docs/dev/architecture.md](docs/dev/architecture.md) — コード構成・学習パイプライン・損失の数式・checkpoint 形式
- [docs/dev/roadmap.md](docs/dev/roadmap.md) — アルゴリズム改善ロードマップ（Phase 0–3）の現状と実験プロトコル

## 構成

- `libriichi/` — Rust コア（ゲームエンジン、mjai ログパーサ、特徴量エンコーダ、アリーナ、統計）。Python から `libriichi` モジュールとして import される
- `mortal/` — Python の学習・推論コード一式（詳細は architecture.md の表）
- `docs/src/` — **公開 mdbook**。内部メモを置かない（内部メモは `docs/dev/`）

## 環境とコマンド

- Windows 11 / PowerShell。conda env `mortal`（Python 3.12 + torch + toml/tqdm/tensorboard）
- libriichi ビルド: `cargo build -p libriichi --lib --release` → `cp target/release/riichi.dll mortal/libriichi.pyd`
- Rust テスト: `cargo test --workspace --no-default-features --features flate2/zlib`
- 学習系の実行はすべて `mortal/` をカレントにして行う（`libriichi.pyd` と `config.toml` がそこにあるため）
- config は env `MORTAL_CFG`（デフォルト `./config.toml`）で指定。**実 config はコミットしない**。リポジトリにあるのは `mortal/config.example.toml` のみ
- オフライン学習: `python train.py`（`[control] online = false`）
- オンライン学習: `python server.py`（バッファ）+ `python client.py`（自己対戦ワーカー、複数可・LAN 可）+ `python train.py`（`online = true`）
- Python 構文チェック: `python -c "import ast; ast.parse(open(r'mortal\train.py', encoding='utf-8').read())"`

## 鉄則

1. **main は稼働中の学習環境**。アルゴリズム実験は `exp/*` ブランチで行い、評価ゲートを通ったものだけ main にマージする
2. **新しい config キーは必ず `.get()` + デフォルト値**で読み、デフォルトは旧挙動を bit 同一に再現する。新機能はデフォルトオフ。理由: checkpoint に config dict が埋め込まれ、`build_engine_from_state`（mortal/engine.py）が古い checkpoint の config を読むため
3. **checkpoint（state_file）のフォーマットを変えない**。やむを得ず変える場合は `build_engine_from_state` と train.py の resume 処理の両方を後方互換に保つ
4. **新しい config キーは `mortal/config.example.toml` にコメント付きで文書化する**（このフォークの慣習）
5. **学習改善の効果判定は test_play の標準誤差基準**（TensorBoard `test_play/avg_ranking_se`）。3000 試合で SE ≈ 0.02。SE 未満の差はノイズとして扱う（`[test_play] gate_sigma` 参照）

## 既知の問題・落とし穴

- train.py: オンラインモードで test_play 後にプロセスがハングする既知バグ → `sys.exit(0)` で子プロセスごと再起動する設計（`main()` が監視ループ）。修正できれば反復が速くなる（原因未特定）
- dataloader.py: GRP の構築を `__init__` でなく `build_iter` で行うのは Windows の DataLoader worker 対策。`__init__` に戻すと壊れる
- `test_play.log_dir` は評価のたびに `rmtree` される。**トレーナー2本でこのディレクトリを共有してはいけない**（A/B 実験時は state_file / best_state_file / tensorboard_dir / test_play.log_dir をすべて分離）
- `config.example.toml` の `[online.server] sample_reuse_rate / sample_reuse_threshold` は**現在未実装**（キーだけ存在。Phase 2 で復活予定、roadmap.md 参照）
- DQN は dueling 構造で、`a_mean` を合法手上で取るため「合法手の Q 平均 ≡ V ヘッド出力」が厳密に成り立つ（AWBC のベースラインはこれを利用）
