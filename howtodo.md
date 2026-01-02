# AI英会話学習システム 構築ガイド

## 概要
GeForce RTX 5090 Laptop 16GBを活用した、音声ベースのAI英会話学習システムの構築方法

## システム要件

### ハードウェア
- **GPU**: GeForce RTX 5090 Laptop 16GB
- **メモリ**: 16GB以上推奨
- **ストレージ**: 50GB以上の空き容量（モデルファイル用）

### ソフトウェア
- **OS**: Windows 10/11
- **Python**: 3.10以上
- **CUDA**: 12.x以上（NVIDIA公式サイトからインストール）

## システムアーキテクチャ

### 主要コンポーネント

1. **音声認識（Speech-to-Text）**
   - **推奨**: Faster-Whisper（OpenAI Whisperの高速版）
   - GPU対応で高速・高精度
   - リアルタイム処理が可能

2. **大規模言語モデル（LLM）**
   - **推奨オプション**:
     - **Llama 3.1 8B Instruct** (VRAM: 約6-8GB)
     - **Mistral 7B Instruct** (VRAM: 約5-7GB)
     - **Phi-3 Medium** (VRAM: 約8-10GB)
   - 16GB VRAMで十分に動作可能
   - 英語会話に特化した指示が可能

3. **音声合成（Text-to-Speech）**
   - **推奨**: Coqui TTS または Piper TTS
   - GPU対応で自然な音声生成
   - 英語ネイティブ音声モデル使用

4. **翻訳機能**
   - **推奨**: LLMに翻訳プロンプトを使用
   - または軽量な翻訳モデル（Helsinki-NLP/opus-mt）

5. **UI フレームワーク**
   - **推奨**: Gradio または Streamlit
   - Pythonで簡単に実装可能
   - リアルタイム更新対応

## 技術スタック

### 必要なPythonライブラリ

```bash
# 基本ライブラリ
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# LLM関連
pip install transformers accelerate bitsandbytes

# 音声認識
pip install faster-whisper

# 音声合成
pip install TTS

# UI
pip install gradio

# 音声処理
pip install sounddevice soundfile numpy

# その他
pip install sentencepiece protobuf
```

## システム設計

### データフロー

```
[マイク入力] 
    ↓
[音声認識 (Faster-Whisper)]
    ↓
[テキスト表示 + 単語分析]
    ↓
[LLM処理 (Llama/Mistral)]
    ↓
[応答テキスト表示 + 翻訳]
    ↓
[音声合成 (TTS)]
    ↓
[スピーカー出力]
```

### UI設計

#### 画面レイアウト
```
┌─────────────────────────────────────────┐
│  AI英会話学習システム                      │
├─────────────────────────────────────────┤
│  [🎤 マイク] [⌨️ キーボード入力]           │
├─────────────────────────────────────────┤
│  会話履歴:                                │
│  ┌───────────────────────────────────┐  │
│  │ You: Hello, how are you?          │  │
│  │ 📖 [日本語訳] [単語]                │  │
│  │                                   │  │
│  │ AI: I'm doing great, thank you!   │  │
│  │ 📖 [日本語訳] [単語]                │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│  設定:                                   │
│  □ 日本語訳を自動表示                     │
│  □ 難しい単語をハイライト                 │
│  □ 発音評価を表示                        │
├─────────────────────────────────────────┤
│  統計: 会話時間: 10分 | 単語数: 150       │
└─────────────────────────────────────────┘
```

## 実装手順

### ステップ1: 環境構築

1. **CUDA環境の確認**
```bash
nvidia-smi
```

2. **Pythonプロジェクトの作成**
```bash
mkdir ai_english_conversation
cd ai_english_conversation
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **依存関係のインストール**
```bash
pip install -r requirements.txt
```

### ステップ2: 各コンポーネントの実装

#### 2.1 音声認識モジュール (speech_recognition.py)

```python
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import soundfile as sf

class SpeechRecognizer:
    def __init__(self, model_size="base"):
        # GPU使用、compute_type="float16"で高速化
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    def transcribe(self, audio_data, sample_rate=16000):
        # 音声データをテキストに変換
        segments, info = self.model.transcribe(audio_data, language="en")
        text = " ".join([segment.text for segment in segments])
        return text
```

#### 2.2 LLMモジュール (llm_handler.py)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMHandler:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # メモリ効率化
        )
        
        # 英会話教師としてのシステムプロンプト
        self.system_prompt = """You are a friendly English conversation teacher. 
        Have natural conversations with the student to help them improve their English.
        Use clear, natural English at an intermediate level.
        Be encouraging and patient."""
    
    def generate_response(self, user_input, conversation_history=[]):
        # 会話履歴を含めてプロンプト作成
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # トークン化して生成
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=150, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
```

#### 2.3 音声合成モジュール (text_to_speech.py)

```python
from TTS.api import TTS

class TextToSpeech:
    def __init__(self):
        # 英語の自然な音声モデル
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)
    
    def speak(self, text, output_path="output.wav"):
        self.tts.tts_to_file(text=text, file_path=output_path)
        return output_path
```

#### 2.4 翻訳・単語分析モジュール (translation.py)

```python
from transformers import pipeline
import spacy

class TranslationHelper:
    def __init__(self, llm_handler):
        self.llm = llm_handler
        # 英語の難易度分析用
        self.nlp = spacy.load("en_core_web_sm")
    
    def translate_to_japanese(self, english_text):
        # LLMを使用して翻訳
        prompt = f"Translate the following English to Japanese: '{english_text}'"
        translation = self.llm.generate_response(prompt, [])
        return translation
    
    def extract_difficult_words(self, text, difficulty_threshold=5):
        # 難しい単語を抽出（CEFR レベルなどで判定）
        doc = self.nlp(text)
        difficult_words = []
        
        for token in doc:
            if token.is_alpha and len(token.text) > 6:  # 簡易的な判定
                difficult_words.append({
                    "word": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_
                })
        
        return difficult_words
```

#### 2.5 メインアプリケーション (app.py)

```python
import gradio as gr
import sounddevice as sd
import numpy as np
from speech_recognition import SpeechRecognizer
from llm_handler import LLMHandler
from text_to_speech import TextToSpeech
from translation import TranslationHelper

class EnglishConversationApp:
    def __init__(self):
        self.recognizer = SpeechRecognizer(model_size="base")
        self.llm = LLMHandler()
        self.tts = TextToSpeech()
        self.translator = TranslationHelper(self.llm)
        self.conversation_history = []
        self.is_recording = False
        self.audio_buffer = []
    
    def process_audio_input(self, audio):
        # 音声認識
        user_text = self.recognizer.transcribe(audio)
        
        # LLM応答生成
        ai_response = self.llm.generate_response(user_text, self.conversation_history)
        
        # 会話履歴に追加
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # 音声合成
        audio_path = self.tts.speak(ai_response)
        
        return user_text, ai_response, audio_path
    
    def create_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎓 AI英会話学習システム")
            
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(source="microphone", type="numpy", label="🎤 マイク入力")
                    text_input = gr.Textbox(label="⌨️ テキスト入力", placeholder="または、ここに入力...")
                    submit_btn = gr.Button("送信", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(label="会話履歴", height=400)
                    audio_output = gr.Audio(label="🔊 AI音声", autoplay=True)
            
            with gr.Row():
                show_translation = gr.Checkbox(label="日本語訳を表示", value=False)
                show_words = gr.Checkbox(label="難しい単語を表示", value=False)
            
            with gr.Row():
                translation_box = gr.Textbox(label="📖 日本語訳", visible=False)
                words_box = gr.Textbox(label="📚 難しい単語", visible=False)
            
            # イベントハンドラ
            def process_conversation(audio, text, history, show_trans, show_word):
                if audio is not None:
                    user_text, ai_response, audio_path = self.process_audio_input(audio)
                else:
                    user_text = text
                    ai_response = self.llm.generate_response(user_text, self.conversation_history)
                    audio_path = self.tts.speak(ai_response)
                
                history.append((user_text, ai_response))
                
                translation = ""
                words = ""
                
                if show_trans:
                    translation = self.translator.translate_to_japanese(ai_response)
                
                if show_word:
                    difficult = self.translator.extract_difficult_words(ai_response)
                    words = "\n".join([f"• {w['word']} ({w['pos']})" for w in difficult])
                
                return history, audio_path, translation, words
            
            submit_btn.click(
                process_conversation,
                inputs=[audio_input, text_input, chatbot, show_translation, show_words],
                outputs=[chatbot, audio_output, translation_box, words_box]
            )
            
            # チェックボックスで表示切り替え
            show_translation.change(
                lambda x: gr.update(visible=x),
                inputs=[show_translation],
                outputs=[translation_box]
            )
            
            show_words.change(
                lambda x: gr.update(visible=x),
                inputs=[show_words],
                outputs=[words_box]
            )
        
        return demo

def main():
    app = EnglishConversationApp()
    demo = app.create_ui()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
```

### ステップ3: 実行

```bash
python app.py
```

ブラウザで `http://127.0.0.1:7860` にアクセス

## VRAM使用量の最適化

RTX 5090 Laptop 16GBでの推奨設定:

| コンポーネント | モデル | VRAM使用量 |
|--------------|--------|-----------|
| Whisper | base | 約1GB |
| LLM | Llama 3.1 8B (8bit) | 約8GB |
| TTS | Tacotron2 | 約2GB |
| **合計** | | **約11GB** |

残り5GBはバッファとして使用可能

### さらなる最適化オプション

1. **4bit量子化**: `load_in_4bit=True` でVRAM使用量を半減
2. **小型モデル**: Phi-3 Mini (3.8B) で約4GB
3. **モデルのオフロード**: CPU/GPUハイブリッド実行

## 追加推奨機能

### 1. 発音評価システム
- **実装**: Whisperの信頼度スコアを使用
- 発音の正確性をフィードバック

### 2. 会話トピック選択
- ビジネス、旅行、日常会話などのシナリオ選択
- トピック別の語彙学習

### 3. 進捗トラッキング
- 会話時間、使用単語数の記録
- 学習履歴のグラフ表示

### 4. スピード調整
- AIの話す速度を調整可能
- リスニング難易度のカスタマイズ

### 5. 文法チェック
- ユーザーの発言を文法チェック
- 改善提案の表示

### 6. シャドーイング練習モード
- AIの発言を繰り返す練習
- 発音比較機能

### 7. 単語帳機能
- 学習した単語を自動保存
- 復習機能の実装

### 8. 会話シナリオ保存
- 過去の会話を保存・復習
- お気に入り会話のブックマーク

## トラブルシューティング

### CUDAエラーが出る場合
```bash
# PyTorchのバージョン確認
python -c "import torch; print(torch.cuda.is_available())"
```

### メモリ不足エラー
- モデルサイズを小さくする
- 4bit量子化を使用
- バッチサイズを1に設定

### 音声認識が遅い場合
- Whisperのモデルを"tiny"または"base"に変更
- `compute_type="int8"`で高速化

### 音声が出ない場合
- オーディオデバイスの設定確認
- sounddeviceのデバイスリスト確認

## パフォーマンス目標

- **音声認識**: 1秒以内
- **LLM応答生成**: 2-3秒
- **音声合成**: 1-2秒
- **合計レスポンス時間**: 5秒以内

## セキュリティとプライバシー

- すべての処理はローカルで実行
- 会話データは外部に送信されない
- モデルは初回起動時にダウンロード

## 参考リソース

- [Faster-Whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Coqui TTS](https://github.com/coqui-ai/TTS)

## まとめ

このシステムは、RTX 5090 Laptop 16GBで快適に動作する英会話学習環境を提供します。すべての処理がローカルで完結するため、プライバシーも保護されます。段階的に機能を追加していくことで、より充実した学習体験を構築できます。

---

**作成日**: 2025年12月24日
**対象GPU**: GeForce RTX 5090 Laptop 16GB
**推奨Python**: 3.10以上

