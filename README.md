# ei-vo

`ei-vo` は MuJoCo ベースのビジュアライザとテレオペレーションのデモを含むリポジトリです。`examples/demo_mj.py` を利用すると、任意自由度のロボットモデルに合わせてデモ軌道を生成・再生できます。

## セットアップ

```bash
pip install -e .
```

MuJoCo のランタイムが必要です。必要に応じて [公式ドキュメント](https://mujoco.readthedocs.io/) を参照し、`MUJOCO_PY_MJKEY_PATH` などの環境変数を設定してください。

## MuJoCo デモ (`examples/demo_mj.py`)

- `--model` で MJCF ファイルを指定します。
- `--angles` に CSV/NPY/JSON 形式の関節軌道 (shape=(T, DOF)) を指定すると、そのまま再生します。
- 軌道ファイルを指定しない場合は、モデルの関節数を自動検出して安全域内のデモ軌道（ウェイポイントまたはサイン波）を生成します。
- `--record` を指定すると、再生内容を MP4 で保存できます（ファイルパスまたはディレクトリを指定可能）。
- `--recordFps` / `--recordSize` を使うと、録画のフレームレートや解像度を調整できます。

例: 3 自由度モデルをウェイポイントデモで再生する

```bash
python examples/demo_mj.py \
  --model tests/models/three_dof_arm.xml \
  --demo wp \
  --hz 240
```

角度ファイル（CSV など）を使う場合:

```bash
python examples/demo_mj.py \
  --model tests/models/three_dof_arm.xml \
  --angles my_angles.csv \
  --deg  # CSV が度[deg]単位の場合
```

## サンプルモデル

- `tests/models/three_dof_arm.xml`: テストやデモに利用できる 3 自由度のアームモデルです。

## テスト

```bash
pytest
```

## ライセンス

プロジェクトのライセンスは `pyproject.toml` を参照してください。
