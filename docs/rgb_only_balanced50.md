# RGB版：各タスク50エピソードへの増量

状態：データ生成・追加学習を実行中。以下の件数は目標であり、完了数は収集結果で確認する。

## データ

| 分割 | 水滴回収 | ペン把持 | キャップ装着 |
|---|---:|---:|---:|
| 学習 | 50 | 50 | 50 |
| 検証 | 10 | 10 | 10 |
| 最終の推論評価 | 5 | 5 | 5 |

既存の完全な成功記録を再利用し、不足分をGazeboの教師制御で生成する。
初期位置は設定値から各軸±5mm。ペンには方向を変えた並進速度2〜5mm/s、角速度0.15〜0.30rad/sを与える。
水滴の上向き速度は4〜8mm/s。学習用の新規試行では10%の指令に標準偏差0.003radの小さな関節誤差を加え、教師による修正動作も記録する。
検証用に指令誤差は加えない。

教師は物体の真値と逆運動学を使って関節目標を作る。モデルが受け取る入力はRGB画像3枚、関節角12値、関節速度12値、言語指示に限定する。
新規収集では深度センサーを無効にする。再利用元の深度ファイルはモデルへ読み込まない。

収集の失敗、起動・通信エラー、中断した試行は保持するが、成功件数および学習対象には含めない。
成功したエピソードID順で学習50件、検証10件を各タスクから選び、画像の存在、有限な関節値、指示文、シードの重複を監査する。
並列実行による目標数を超えた記録も、選択対象の外に保持する。

- 収集先：`datasets/rgb_only_balanced50_v1/`
- 固定した学習・検証用データ：`datasets/rgb_only_balanced50_v1_selected/`
- 選択とファイルハッシュ：選択先の `selection_manifest.json`
- 事前に決めたシード：収集先の `collection_plan.json`

## 学習と評価

各30件が揃った時点で90件の先行学習を行い、各50件と検証各10件が揃った後に150件でさらに追加学習する。
SmolVLA本体のLoRA・入出力射影層を更新する。関節目標は直接出力する。
検証誤差でチェックポイントを選び、必要に応じて同じ特徴を用いるGRU出力層も学習・比較する。

開発シード6400000・6400001で候補を比較する場合は、まずタスク別成功数の最小値、次に合計成功数で選ぶ。同率なら構成の単純なSmolVLA本体の出力を選ぶ。
最終評価は別のシード6300000〜6300004で各タスク5回ずつ行う。
最終評価の結果を見て学習データやモデル選択を変更しない。

## 再実行用スクリプト

Ubuntu 24.04で `rgb-only-smolvla` ブランチに切り替え、ROS 2環境を読み込んで実行する。

```bash
git switch rgb-only-smolvla
source /opt/ros/jazzy/setup.bash

# 既存フォルダがあれば成功件数を確認して不足分だけを収集する。
# 異なるプロセスを並行実行する場合は、ROS_DOMAIN_IDとシード範囲を分ける。
ROS_DOMAIN_ID=90 .venv-vla/bin/python collect_rgb_dataset.py \
  --output datasets/rgb_only_balanced50_v1 --task task1

# 成功各50件・検証各10件が揃った後、新しい選択先へ固定する。
.venv-vla/bin/python prepare_rgb_dataset.py \
  --source datasets/rgb_only_balanced50_v1 \
  --output datasets/rgb_only_balanced50_v1_selected

# 固定したチェックポイントの最終評価。出力先は新しい名前を指定する。
ROS_DOMAIN_ID=90 .venv-vla/bin/python evaluate_rgb_policy.py \
  --checkpoint artifacts/smolvla_rgb_only_balanced50_selected \
  --output runs/rgb_only_balanced50_test_manual \
  --split test --record-first
```

この実験の変更はRGB用ブランチに限定し、深度ありの `main` は変更しない。
