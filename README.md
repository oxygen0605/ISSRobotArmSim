# ISSRobotArmSim — SO-101 × 2 / SmolVLA

Ubuntu 24.04・ROS 2 Jazzy・Gazebo Harmonicで動く、ISS船内をイメージした双腕ロボットのシミュレーションです。
同じ学習済みSmolVLAと共有出力部を使用し、英語の指示に応じて3タスクを実行します。

| タスク | 内容 |
|---|---|
| Task 1 | 右腕で把持したスポンジを、両腕の間を漂う水滴に接触させて回収 |
| Task 2 | 両腕の間で回転するペンを左腕で把持して保持 |
| Task 3 | 左腕でペン、右腕でキャップを把持し、軸合わせ・挿入を実行 |

両腕は作業台の同じ一辺に並び、一人称カメラを両腕の間に設置しています。
モデル入力はカメラ画像・RGB-Dから推定した幾何情報・関節状態・指示文です。
シミュレータの物体真値と接触情報は初期化・成功判定に使用します。

## 動作環境とセットアップ

- Ubuntu 24.04（WSL 2 + WSLgで動作確認）
- CUDAを利用できるNVIDIA GPUと対応ドライバー
- ROS 2 Jazzy / Gazebo Harmonic
- Python 3.12
- 初回セットアップにはインターネット接続が必要

~~~bash
git clone https://github.com/oxygen0605/ISSRobotArmSim.git
cd ISSRobotArmSim
sudo bash scripts/install_ros_jazzy.sh
bash scripts/setup_runtime.sh
~~~

セットアップはPython環境、固定バージョンの基盤モデル・プロセッサ、水滴描画プラグイン、ROSパッケージを準備します。
基盤モデルとプロセッサは約912 MBあり、Gitには含めず公開元から取得します。
本課題で追加学習した重みは artifacts/smolvla_one_policy/ に含めます。
学習元と固定revisionは scripts/fetch_vla_models.sh と THIRD_PARTY_NOTICES.md に記載しています。

## 実行

新しいターミナルでリポジトリへ移動して実行します。

~~~bash
source scripts/ros_env.sh
ros2 launch iss_capture_ros one_policy.launch.py task:=all seed:=3300002 record:=true
~~~

個別実行は task:=task1・task2・task3、GUIなしは gui:=false、録画なしは record:=false です。
初期配置と英語の指示文は config/task_scenes.json、使用する重みは checkpoint:=... で指定できます。

録画と結果は runs/one_policy_日時/taskN/ に保存します。

- policy_front.mp4：一人称視点
- policy_side.mp4：側面視点
- policy_overview.mp4：俯瞰視点
- policy_opposite.mp4：机の反対側の視点
- result.json：成功判定・条件・モデルのハッシュ
- trace.json：各ステップの指令など

俯瞰・対面カメラは記録用です。推論には一人称・側面・過去の一人称画像を使用します。

## 公開データセット・学習・オフライン推論

[Hugging Face: oxygen65/ISSRobotArm](https://huggingface.co/datasets/oxygen65/ISSRobotArm) で、
学習24・検証8エピソード（1,859時刻、約1.81 GB）のRGB-D教師データを公開しています。
PNG/NPZ/JSONによる独自形式で、LeRobotDataset形式ではありません。
config/dataset_source.jsonに固定revisionを記録し、取得時にアーカイブと展開後ファイルのSHA-256を検証します。

~~~bash
source scripts/ros_env.sh
python scripts/download_dataset.py
bash scripts/train_task_scenes.sh
~~~

学習コマンドは公開モデルを初期値として追加学習し、新しいartifacts/retrained_日時/へ保存します。
引数で別の新規出力先を指定できます。元の公開モデルは置き換えません。
学習スクリプトを直接実行する場合も、標準のデータフォルダがなければ公開データを取得します。
公開モデルは以前の学習済みアダプターも引き継いでいるため、過去の全学習履歴を再現するコマンドではありません。

公開データの入力画像・深度・関節状態だけで、物理シミュレーションを動かさず推論できます。

~~~bash
python infer_dataset.py --episode validation_task1_4200000 --max-frames 10
python infer_dataset.py --episode validation_task2_4200001 --max-frames 10
python infer_dataset.py --episode validation_task3_4200001 --max-frames 10
~~~

指定エピソードだけを取得し、runs/dataset_inference_日時/へ予測を保存します。
教師指令・物体真値は推論入力に渡しません。観測の再生であり、閉ループの成功率測定とは区別します。
Gazeboのオンライン推論はライブ画像を使うため、データセットのダウンロードは不要です。

新しいモデルは checkpoint:=/absolute/path/to/new/head/best を指定してGazeboで評価できます。
初期条件を分けた評価用に evaluate_task_scene_policy.py、データ再生成用に collect_task_scene.py / collect_task_scene_suite.py も収録しています。

## この公開版の範囲

Task 1〜3の学習・推論・シミュレーション・録画に必要なソース、設定、アセット、最終学習済み重みと、
セットアップ手順・ライセンス情報を収録しています。
学習・評価・データ生成スクリプトを含みます。データセット本体、過去のチェックポイント、録画済み動画、ビルド成果物、仮想環境はGitに含めません。
実行ファイルから参照される補助モジュールは保持しています。

開発環境で固定モデルを評価した結果は、各タスク4条件でTask 1が4/4、Task 2が4/4、Task 3が3/4でした。
少数条件の結果であり、すべての初期状態で成功することを保証するものではありません。
Task 1の深度計測には理想化した表面を使い、水滴の吸着は接触後の球体追従として描画します。
吸水・流体計算は行いません。

## ライセンス

自作コード・自作アセットは [Apache License 2.0](LICENSE) で公開します。
第三者のコード、ロボットモデル、学習元モデル等はそれぞれのライセンスに従います。
出典と再配布条件は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) と licenses/ を参照してください。
SO-101は使用するURDFとその参照先13メッシュに絞り、元のライセンス・出典を添付しています。
参照写真や第三者のISS内装アセット、フォント本体は同梱していません。

本プロジェクトは独立した教育用シミュレーションであり、JAXA等の公式製品ではありません。
