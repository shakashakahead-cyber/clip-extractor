クリップ抽出くん ベータ (Clip Extractor Beta)
===========================================

動画の音声を解析し、盛り上がっている箇所をハイライト候補として抽出するアプリです。

主な機能
--------
- AI 音声解析による候補抽出
- スコアフィルタ付き候補一覧
- 開始/終了時刻の手動編集
- 個別 / 連結エクスポート
- 候補再生時に終了時刻で自動停止

現在の解析方式
--------------
- モデル: EfficientAT (mn10_as)
- 推論: ONNX Runtime + DirectML (DmlExecutionProvider)
- 音声: 32kHz mono、10秒窓、2.5秒ホップ

チューニング方針:
- 笑い優先
- Cheering/Applause/Clapping を重視
- Crowd/Screaming/Shouting は重みを下げて誤検出抑制
- 動的区間境界 + 重複抑制 (NMS)

スコアについて
--------------
- 一覧表示の score は、候補間で比較しやすいよう再キャリブレーションされた表示スコアです。
- 内部の生スコアは details["score_raw"] に保持されます。

動作要件
--------
- Windows
- Python 3.10 以上推奨
- DirectX 12 対応 GPU (AMD / NVIDIA)
- FFmpeg

重要:
- 現在は DirectML 必須です。
- DmlExecutionProvider が使えない環境ではエラーになります。
- CPU 自動フォールバックはありません。

セットアップ
------------
1. リポジトリを取得
2. 依存関係をインストール
3. アプリ起動

例:
git clone https://github.com/shakashakahead-cyber/clip-extractor.git
cd clip-extractor
pip install -r requirements.txt
python main.py

初回解析時は EfficientAT の重み取得/ONNX 変換で時間がかかる場合があります。

FFmpeg
------
ffmpeg.exe を次のどちらかで利用可能にしてください。
- PATH に追加
- アプリ実行フォルダに配置

出力
----
- 個別: base_001.mp4, base_002.mp4, ...
- 連結: 単一の mp4

ライセンス
----------
MIT License
サードパーティ情報は THIRD_PARTY_NOTICES.txt を参照してください。

