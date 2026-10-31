# smolvla_rgb_only

公開用最終モデル（30,000 optimizer step）。基盤モデルは `lerobot/smolvla_base`、入力はRGB画像3枚・関節角12値・速度12値・言語です。深度や物体真値は入力しません。

- アダプターSHA-256：`4a2aa37146f053aa5b53639dd23295228467fa3e818b03945e32329d73454986`
- 保存時の更新数：30000
- チャンク長：50／使用：10
- 公開データ：oxygen65/ISSRobotArm（revision `f9df1f800ea838a9d1baf21ffd4d9d87e1355b64`）

基盤モデルはセットアップ時に取得します。このフォルダには追加学習したLoRA・入出力層の重みを保存し、optimizer状態は含めません。初期モデルの学習履歴は公開150エピソードだけに限定されません。ライセンスはリポジトリのLICENSEとTHIRD_PARTY_NOTICES.mdを参照してください。
