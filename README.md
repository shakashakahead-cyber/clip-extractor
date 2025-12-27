# クリップ抽出くん ベータ (Clip Extractor Beta)

動画内の「盛り上がり（笑い声・歓声）」をAIが自動検知し、ハイライトシーンを抽出・切り抜きするデスクトップアプリです。

## ダウンロード

[Releases](https://github.com/shakashakahead-cyber/clip-extractor/releases) からインストーラをダウンロードしてください。

## 必要条件

### FFmpeg（必須）
本ソフトを使用するには「FFmpeg」が必要です。

1. [FFmpeg公式サイト](https://www.gyan.dev/ffmpeg/builds/) からWindows用ビルドをダウンロード
2. 解凍して `ffmpeg.exe` を取り出す
3. 以下のいずれかの方法で設定：
   - システム環境変数のPathに追加（推奨）
   - アプリと同じフォルダに配置

## 使い方

1. アプリを起動
2. 「動画を選択」ボタンで解析したい動画を選択
3. AI解析が自動実行され、ハイライト候補がリスト表示
4. 必要なシーンにチェック → 「書き出し」で保存

## GPU対応

AMD/NVIDIAグラフィックボードを搭載している場合、自動的にGPUを使用して高速に解析します。

## 開発者向け

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/shakashakahead-cyber/clip-extractor.git
cd clip-extractor

# 依存関係をインストール
pip install -r requirements.txt

# ONNXモデルをダウンロード（別途配布）
# Cnn14_batch32.onnx, Cnn14_batch32.onnx.data を配置

# 実行
python main.py
```

### 必要なファイル（別途ダウンロード）
- `Cnn14_batch32.onnx` / `.onnx.data` - PANNs音声分類モデル
- `ffmpeg.exe` - FFmpeg実行ファイル

## ライセンス

MIT License

本ソフトはオープンソースコンポーネントを使用しています。
詳細は [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) を参照してください。
