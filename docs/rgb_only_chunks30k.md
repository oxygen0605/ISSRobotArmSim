# RGB SmolVLA：チャンク50・30,000 optimizer step

2026-10-31の指定に基づく実験。`rgb-only-smolvla` ブランチで実行し、深度ありの `main` と過去の重み・評価は保持する。

| 設定 | 値 |
|---|---|
| 学習データ | 各タスク50件、計150件・9,286フレーム |
| 検証データ | 各タスク10件、計30件・1,793フレーム |
| 学習開始点 | `smolvla_rgb_only_balanced50_selected` の重み。optimizerは新規 |
| Optimizer更新数 | 30,000。早期終了なし |
| 実効バッチ | 64 |
| Micro batch / 累積 | 32 / 2回 |
| 学習率 | 1,000更新で0から3e-4へ線形warmup、その後cosine decay、30,000更新で3e-5 |
| 保存する更新数 | 10,000 / 20,000 / 30,000 |
| チャンク長 / 実行長 | 50 / 10 |
| 動作間隔 | 0.1秒。5秒分を予測し、先頭1秒分を実行して再推論 |
| 最適化 | AdamW、weight decay 1e-4、勾配norm上限1、bfloat16 |
| 学習方式 | 既存のdeterministic endpointを継続。LoRA r16 / alpha32 / dropout0.05 |
| 補助損失 | 学習時のみのロボット運動学損失、重み0.003 |
| 入力 | RGB 256×256を3枚、関節角12値、関節速度12値、言語 |
| 出力 | 左右12軸の絶対関節目標。深度・物体真値は入力しない |

開始点の重みには以前の学習履歴が含まれる。この実験で読み込む学習データは、固定済みの150件だけである。

エピソード末尾を越えるチャンクは配列上では最後の教師目標で埋めるが、実在しない時刻は学習・検証の損失から除外する。未来目標が別エピソードへまたがることはない。累積時は有効な目標数に応じて各micro batchの損失を重み付けし、一括バッチ64と同じ目的関数になるようにする。

1,000更新ごとに各タスク最大128検証フレームから50ステップを予測する。選択用スコアは実際に使う先頭10ステップの関節・グリッパー・運動学の誤差で計算する。`best` は検証スコア最良の参考重みであり、指定された30,000更新の重みは別の `step_30000` に必ず保存する。`latest` には1,000更新ごとの再開状態も保存する。

## 実行

```bash
git switch rgb-only-smolvla
bash scripts/train_rgb_chunks30k.sh
```

同じ出力先を上書きしない。再実験する場合は `RGB_OUTPUT_DIR` に新しいパスを設定する。

GPUメモリに合わせてmicro batchを変更しても、実効バッチは64のまま保つ。

```bash
RGB_MICRO_BATCH_SIZE=16 RGB_OUTPUT_DIR=artifacts/smolvla_rgb_chunks50_steps30000_other \
  bash scripts/train_rgb_chunks30k.sh
```

出力先：`artifacts/smolvla_rgb_chunks50_steps30000_v1/`

- `training_config.json`：実行設定、ソース・データのハッシュ。
- `progress.json`：optimizer step、実際の学習率、損失、GPU使用量、完了状態。
- `step_10000/`、`step_20000/`、`step_30000/`：指定チェックポイント。
- 各保存先の `training_state.pt`：Adam・CPU/CUDA/NumPy乱数・更新数を含むローカル再開状態。
- `validation_history.json`：検証履歴。

途中再開では元の引数を維持し、`--resume` の代わりに `--continue-training <出力先>/latest` を渡す。更新数は0に戻らず、cosine scheduleも続きから再開する。途中からmicro batchを変更する再開は設定不一致として拒否する。

この学習実行だけで成功率の改善を主張しない。過去の未知条件テストと学習条件での診断結果はそのまま保持し、再評価は別の実験記録として扱う。
