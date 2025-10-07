# Examples

`examples` ディレクトリには Panda ロボットの MuJoCo モデルを使ったデモスクリプトが入っています。ここでは `demo_mj.py` の使い方と、軌道の準備・録画機能について説明します。

## 前提条件

- MuJoCo 2.x がインストールされ、`LD_LIBRARY_PATH` や `MUJOCO_PY_MJKEY_PATH` 等の環境変数が設定済みであること
- Panda の MJCF (`panda.xml`) が手元にあること
- Python 3.9 以降、および `requirements.txt` に記載の依存関係がインストールされていること

## デモの実行

角度ファイルを指定しない場合、スクリプトは内蔵のデモ軌道を生成して再生します。以下はウェイポイントデモを実時間で再生する例です。

```bash
python examples/demo_mj.py --model /path/to/panda.xml
```

### オプション

| オプション | 説明 |
| --- | --- |
| `--angles PATH` | CSV / NPY / JSON 形式の関節角度ファイルを読み込みます (`shape=(T,7)`) |
| `--deg` | 角度ファイルが度数法 [deg] のときに指定します (ラジアンに変換) |
| `--hz FLOAT` | 再生周波数を Hz で指定します (デフォルト: 240.0) |
| `--loop` | 再生をループさせます |
| `--demo {wp,sine}` | 角度ファイル未指定時のデモ軌道を切り替えます |
| `--segT FLOAT` | ウェイポイントデモの区間時間 [s] (デフォルト: 1.5) |
| `--slow FLOAT` | 再生をスロー再生します (`>1` でゆっくり) |
| `--record PATH` | 指定すると録画動画を保存します (例: `output.mp4`) |
| `--recordFps FLOAT` | 録画動画のフレームレートを明示的に指定します |
| `--recordSize W H` | 録画動画のサイズ (幅, 高さ) をピクセル単位で指定します |

## 録画付きの実行

録画したい場合は `--record` を指定してください。フレームレートや解像度を変更する場合は `--recordFps` と `--recordSize` を併用します。

```bash
python examples/demo_mj.py \
    --model /path/to/panda.xml \
    --record demo.mp4 \
    --recordFps 60 \
    --recordSize 1920 1080
```

録画時も通常のビューワ表示は保持され、終了時に動画ファイルが保存されます。

## 軌道ファイルの準備

`--angles` で読み込むファイルは 7 関節の角度列を表す 2 次元配列です。サンプルとして、CSV を NumPy で生成するコード例を以下に示します。

```python
import numpy as np

# shape = (T, 7)
angles = np.linspace(0, 1, 240)[:, None] * np.ones((1, 7))
np.savetxt("traj.csv", angles, delimiter=",")
```

JSON や NPY 形式も同様に読み込めます。角度が度数法の場合は `--deg` を忘れずに指定してください。

## テスト

`tests/` にはデモスクリプトの読み込みや録画処理を検証する Pytest ベースのテストが含まれています。以下で実行できます。

```bash
pytest
```

MuJoCo のネイティブライブラリを必要としないようスタブが用意されているため、ローカル環境でもそのままテスト可能です。
