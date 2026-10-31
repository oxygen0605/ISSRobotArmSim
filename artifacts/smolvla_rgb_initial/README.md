# smolvla_rgb_initial

30,000ステップ追加学習の開始点。基盤モデルは `lerobot/smolvla_base`、入力はRGB画像3枚・関節角12値・速度12値・言語です。深度や物体真値は入力しません。

- アダプターSHA-256：`adebaeb9d7ff9195dcf98398841776a468620d3c735c97fa739931f33c7e74f7`
- 保存時の更新数：2500
- チャンク長：1／使用：1
- 公開データ：oxygen65/ISSRobotArm（revision `e96f1ede890c8f02d1800bae28c1456721621958`）

基盤モデルはセットアップ時に取得します。このフォルダには追加学習したLoRA・入出力層の重みを保存し、optimizer状態は含めません。初期モデルの学習履歴は公開150エピソードだけに限定されません。ライセンスはリポジトリのLICENSEとTHIRD_PARTY_NOTICES.mdを参照してください。
