# AI英会話学習システム

GeForce RTX 5090 Laptop 16GBを活用した、音声ベースのAI英会話学習システムです。

## 機能

- 🎤 **音声認識**: マイクからの音声入力を英語テキストに変換
- 💬 **AI会話**: LLM（Llama 3.1 8B）との自然な英語会話
- 🔊 **音声合成**: AIの応答を自然な英語音声で再生
- 📖 **日本語訳**: 会話内容の日本語訳を表示
- 📚 **単語学習**: 難しい単語の抽出と日本語訳を表示
- ⌨️ **テキスト入力**: マイクの代わりにキーボード入力も可能

## システム要件

### ハードウェア
- **GPU**: GeForce RTX 5090 Laptop 16GB（または同等のGPU）
- **メモリ**: 16GB以上推奨
- **ストレージ**: 50GB以上の空き容量（モデルファイル用）

### ソフトウェア
- **OS**: Windows 10/11
- **Python**: 3.10以上（3.13対応）
- **CUDA**: 12.x以上（NVIDIA公式サイトからインストール）

## セットアップ

### 1. CUDA環境の確認

```bash
nvidia-smi
```

CUDAが正しくインストールされていることを確認してください。

### 2. Python仮想環境の作成

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

**注意**: `bitsandbytes`はPython 3.13で動作しない可能性があります。
- Python 3.13を使用している場合: `bitsandbytes`なしで動作します（VRAM使用量が増えます）
- Python 3.12以下の場合: `pip install bitsandbytes`で追加インストール可能

### 4. spaCy英語モデルのインストール

```bash
python -m spacy download en_core_web_sm
```

### 5. アプリケーションの起動

```bash
python app.py
```

ブラウザで `http://127.0.0.1:7860` にアクセスしてください。

## 使い方

1. **マイク入力**: マイクボタンをクリックして音声を録音し、送信ボタンをクリック
2. **テキスト入力**: テキストボックスに英語を入力してEnterキーまたは送信ボタンをクリック
3. **日本語訳の表示**: 「日本語訳を表示」チェックボックスをONにする
4. **単語学習**: 「難しい単語を表示」チェックボックスをONにする
5. **会話履歴のクリア**: 「会話履歴をクリア」ボタンで履歴をリセット

## プロジェクト構造

```
e_conversation/
├── app.py                  # メインアプリケーション
├── speech_recognition.py   # 音声認識モジュール
├── llm_handler.py         # LLM処理モジュール
├── text_to_speech.py      # 音声合成モジュール
├── translation.py         # 翻訳・単語分析モジュール
├── requirements.txt       # 依存関係
├── README.md             # このファイル
└── howtodo.md            # 詳細な構築ガイド
```

## トラブルシューティング

### CUDAエラーが出る場合

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`True`が表示されない場合は、PyTorchのCUDA対応版が正しくインストールされていない可能性があります。

### メモリ不足エラー

- Python 3.13の場合: `bitsandbytes`が使えないため、float16で動作します（VRAM使用量が増えます）
- より小さなモデル（例: `meta-llama/Llama-3.1-3B-Instruct`）を使用
- `llm_handler.py`でモデル名を変更: `model_name="meta-llama/Llama-3.1-3B-Instruct"`

### 音声認識が遅い場合

`speech_recognition.py`の`model_size`を`"tiny"`に変更

### 音声が出ない場合

- オーディオデバイスの設定を確認
- `sounddevice`のデバイスリストを確認: `python -c "import sounddevice as sd; print(sd.query_devices())"`

## ライセンス

このプロジェクトは教育目的で作成されています。

## 参考リソース

- [Faster-Whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Coqui TTS](https://github.com/coqui-ai/TTS)

