# クリップ抽出くん ベータ (Clip Extractor Beta)

動画の音声を解析して、盛り上がっている箇所（ハイライト候補）を抽出するデスクトップアプリです。

## 主な機能
- 動画から音声を抽出して AI 解析
- 候補クリップの一覧表示とスコアフィルタ
- 候補の開始/終了時刻の手動調整
- 候補の個別書き出し / 連結書き出し
- 候補再生時に「候補の終了時刻」で自動停止

## 現在の解析方式
- モデル: EfficientAT (`mn10_as`)
- 推論: ONNX Runtime + DirectML (`DmlExecutionProvider`)
- 音声条件: 32kHz / mono / 10秒窓（2.5秒ホップ）

補足:
- 解析は「笑い」を優先しつつ、`Cheering/Applause/Clapping` を重視します。
- `Crowd/Screaming/Shouting` などは重みを下げて誤検出を抑える調整を入れています。
- 区間境界は固定幅ではなく、スコア推移に基づく動的境界です。
- 近い区間は重複抑制（NMS）で整理します。

## スコアについて
- 一覧に表示される `score` は、フィルタ運用しやすいように候補集合内で再キャリブレーションされた表示スコアです。
- 内部では生スコア（raw）も保持しており、実装上は `details["score_raw"]` で参照できます。

## 動作要件
- OS: Windows
- Python: 3.10 以上推奨
- GPU: DirectX 12 対応 GPU（AMD / NVIDIA）
- FFmpeg: 必須

重要:
- 現在は DirectML 前提です。`DmlExecutionProvider` が使えない環境は起動時/解析時にエラーになります（CPU への自動フォールバックなし）。

## セットアップ
```bash
git clone https://github.com/shakashakahead-cyber/clip-extractor.git
cd clip-extractor
pip install -r requirements.txt
python main.py
```

初回解析時:
- EfficientAT の重み取得と ONNX 変換が走るため、初回のみ時間がかかる場合があります。

## FFmpeg
`ffmpeg.exe` が必要です。以下どちらかで準備してください。
- システム PATH に `ffmpeg.exe` を通す
- アプリ実行フォルダに `ffmpeg.exe` を置く

## 出力
- 個別書き出し: `base_001.mp4`, `base_002.mp4`, ...
- 連結書き出し: 単一の `mp4`

## 既知事項
- Flet の `Video()` は将来版で非推奨予定のため、将来的に `flet-video` への移行が必要です。

## ライセンス
MIT License  
サードパーティ情報は `THIRD_PARTY_NOTICES.txt` を参照してください。

