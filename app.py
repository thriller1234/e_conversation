import gradio as gr
import sounddevice as sd
import numpy as np
import os
import tempfile
from speech_recognition import SpeechRecognizer
from llm_handler import LLMHandler
from text_to_speech import TextToSpeech
from translation import TranslationHelper

class EnglishConversationApp:
    def __init__(self):
        """
        AI英会話学習アプリケーションの初期化
        """
        print("Initializing AI English Conversation System...")
        self.recognizer = SpeechRecognizer(model_size="base")
        self.llm = LLMHandler()
        self.tts = TextToSpeech()
        self.translator = TranslationHelper(self.llm)
        self.conversation_history = []
        self.is_recording = False
        self.audio_buffer = []
        print("System initialized successfully!")
    
    def process_audio_input(self, audio):
        """
        音声入力を処理してAI応答を生成
        
        Args:
            audio: 音声データ（タプル: (sample_rate, audio_array) または ファイルパス）
        
        Returns:
            user_text: ユーザーの発言テキスト
            ai_response: AIの応答テキスト
            audio_path: AI音声ファイルのパス
        """
        if audio is None:
            print("DEBUG process_audio_input: audio is None")
            return "", "", None
        
        print(f"DEBUG process_audio_input: audio type = {type(audio)}")
        
        # ファイルパスの場合
        if isinstance(audio, str):
            import soundfile as sf
            print(f"DEBUG: ファイルパスから読み込み: {audio}")
            audio_data, sample_rate = sf.read(audio)
        else:
            # タプルの場合
            try:
                sample_rate, audio_data = audio
                print(f"DEBUG: タプルから取得 - sample_rate: {sample_rate}, audio_data type: {type(audio_data)}")
            except Exception as e:
                print(f"DEBUG: タプルの展開エラー: {e}")
                return "", "", None
        
        # 音声データが空でないか確認
        if audio_data is None:
            print("警告: 音声データがNoneです")
            return "", "", None
        
        # numpy配列の場合の処理
        import numpy as np
        if isinstance(audio_data, np.ndarray):
            if audio_data.size == 0:
                print("警告: 音声データが空です（size=0）")
                return "", "", None
            print(f"DEBUG: 音声データ shape = {audio_data.shape}, size = {audio_data.size}")
        elif hasattr(audio_data, '__len__'):
            if len(audio_data) == 0:
                print("警告: 音声データが空です（len=0）")
                return "", "", None
        else:
            print(f"警告: 予期しない音声データ形式: {type(audio_data)}")
            return "", "", None
        
        # モノラル音声に変換（ステレオの場合は平均を取る）
        if isinstance(audio_data, np.ndarray) and len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            print(f"DEBUG: ステレオからモノラルに変換、新しいshape = {audio_data.shape}")
        
        # データ型をfloat32に変換（Faster-Whisperはfloat32を期待）
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.int16:
                # int16からfloat32に変換し、-1.0から1.0の範囲に正規化
                audio_data = audio_data.astype(np.float32) / 32768.0
                print(f"DEBUG: int16からfloat32に変換（正規化済み）")
            elif audio_data.dtype != np.float32:
                # その他の型もfloat32に変換
                audio_data = audio_data.astype(np.float32)
                print(f"DEBUG: {audio_data.dtype}からfloat32に変換")
        
        # 音声認識
        try:
            print("DEBUG: 音声認識を開始...")
            user_text = self.recognizer.transcribe(audio_data, sample_rate=sample_rate)
            print(f"DEBUG: 音声認識結果: '{user_text}'")
        except Exception as e:
            import traceback
            print(f"音声認識エラー: {e}")
            print(f"トレースバック: {traceback.format_exc()}")
            return "", "", None
        
        if not user_text or not user_text.strip():
            print("警告: 音声からテキストを認識できませんでした")
            return "", "", None
        
        # LLM応答生成
        ai_response = self.llm.generate_response(user_text, self.conversation_history)
        
        # 会話履歴に追加
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # 音声合成
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = self.tts.speak(ai_response, temp_file.name)
        
        return user_text, ai_response, audio_path
    
    def process_text_input(self, text):
        """
        テキスト入力を処理してAI応答を生成
        
        Args:
            text: ユーザーの入力テキスト
        
        Returns:
            user_text: ユーザーの発言テキスト
            ai_response: AIの応答テキスト
            audio_path: AI音声ファイルのパス
        """
        if not text or not text.strip():
            return "", "", None
        
        user_text = text.strip()
        
        # LLM応答生成
        ai_response = self.llm.generate_response(user_text, self.conversation_history)
        
        # 会話履歴に追加
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # 音声合成
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = self.tts.speak(ai_response, temp_file.name)
        
        return user_text, ai_response, audio_path
    
    def create_ui(self):
        """
        Gradio UIの作成
        
        Returns:
            Gradio Blocksオブジェクト
        """
        with gr.Blocks() as demo:
            gr.Markdown("# 🎓 AI英会話学習システム")
            gr.Markdown("マイクまたはテキストで英語で会話を始めましょう！")
            
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone"], 
                        type="numpy", 
                        label="🎤 マイク入力"
                    )
                    text_input = gr.Textbox(
                        label="⌨️ テキスト入力", 
                        placeholder="または、ここに英語で入力..."
                    )
                    submit_btn = gr.Button("送信", variant="primary", size="lg")
            
            with gr.Row():
                chatbot = gr.Chatbot(
                    label="会話履歴", 
                    height=400,
                    show_label=True
                )
            
            with gr.Row():
                audio_output = gr.Audio(
                    label="🔊 AI音声", 
                    autoplay=True,
                    visible=True
                )
            
            with gr.Row():
                show_translation = gr.Checkbox(
                    label="📖 日本語訳を表示", 
                    value=False
                )
                show_words = gr.Checkbox(
                    label="📚 難しい単語を表示", 
                    value=False
                )
            
            with gr.Row():
                translation_box = gr.Textbox(
                    label="📖 日本語訳", 
                    visible=False,
                    lines=3
                )
                words_box = gr.Textbox(
                    label="📚 難しい単語とその意味", 
                    visible=False,
                    lines=5
                )
            
            # 会話履歴を保存するための状態変数
            conversation_state = gr.State(value=[])
            
            # イベントハンドラ
            def process_conversation(audio, text, history, show_trans, show_word):
                """
                会話を処理する関数
                """
                user_text = ""
                ai_response = ""
                audio_path = None
                translation = ""
                words_info = ""
                
                # デバッグ情報
                print(f"DEBUG: audio type = {type(audio)}, audio = {audio}")
                print(f"DEBUG: text = {text}")
                
                # 音声またはテキスト入力の処理
                # Gradioの新しいバージョンでは、audioはタプル(sample_rate, audio_array)またはNone
                if audio is not None:
                    try:
                        print(f"DEBUG: 音声データを処理中...")
                        if isinstance(audio, tuple) and len(audio) == 2:
                            sample_rate, audio_data = audio
                            print(f"DEBUG: sample_rate = {sample_rate}, audio_data type = {type(audio_data)}, shape = {audio_data.shape if hasattr(audio_data, 'shape') else 'N/A'}")
                            if audio_data is not None:
                                if hasattr(audio_data, '__len__') and len(audio_data) > 0:
                                    user_text, ai_response, audio_path = self.process_audio_input(audio)
                                    print(f"DEBUG: 音声認識結果: user_text = {user_text}")
                                else:
                                    print("DEBUG: 音声データが空です")
                        elif isinstance(audio, str):
                            # ファイルパスの場合
                            print(f"DEBUG: ファイルパスから読み込み: {audio}")
                            user_text, ai_response, audio_path = self.process_audio_input(audio)
                        else:
                            print(f"DEBUG: 予期しない音声データ形式: {type(audio)}")
                    except Exception as e:
                        import traceback
                        print(f"音声処理エラー: {e}")
                        print(f"トレースバック: {traceback.format_exc()}")
                        # エラーが発生した場合はテキスト入力にフォールバック
                        if text and text.strip():
                            print("DEBUG: テキスト入力にフォールバック")
                            user_text, ai_response, audio_path = self.process_text_input(text)
                        else:
                            print("DEBUG: エラーが発生し、処理できませんでした")
                            return history, None, "", "", ""
                
                # テキスト入力の処理
                if not user_text and text and text.strip():
                    print("DEBUG: テキスト入力を処理中...")
                    user_text, ai_response, audio_path = self.process_text_input(text)
                
                if not user_text or not ai_response:
                    print(f"DEBUG: 処理結果が空です。user_text = {user_text}, ai_response = {ai_response}")
                    return history, None, "", "", ""
                
                # 会話履歴に追加（Gradioの新しいフォーマットに対応）
                # 新しいバージョンでは辞書形式が必要
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": ai_response})
                
                # 日本語訳の取得
                if show_trans:
                    translation = self.translator.translate_to_japanese(ai_response)
                
                # 難しい単語の抽出と翻訳
                if show_word:
                    difficult_words = self.translator.extract_difficult_words(ai_response)
                    if difficult_words:
                        word_translations = self.translator.get_word_translations(difficult_words)
                        words_info = "\n".join([
                            f"• {w['word']} ({w['pos']}): {word_translations.get(w['word'], 'N/A')}"
                            for w in difficult_words[:10]  # 最大10個まで表示
                        ])
                    else:
                        words_info = "難しい単語は見つかりませんでした。"
                
                return history, audio_path, translation, words_info
            
            # 送信ボタンのイベント
            submit_btn.click(
                process_conversation,
                inputs=[audio_input, text_input, chatbot, show_translation, show_words],
                outputs=[chatbot, audio_output, translation_box, words_box]
            )
            
            # テキスト入力のEnterキーイベント
            text_input.submit(
                process_conversation,
                inputs=[audio_input, text_input, chatbot, show_translation, show_words],
                outputs=[chatbot, audio_output, translation_box, words_box]
            )
            
            # 音声入力の変更イベント（録音完了時に自動処理）
            def on_audio_change(audio, history, show_trans, show_word):
                """音声が変更された時（録音完了時）に自動的に処理"""
                if audio is None:
                    return history, None, "", ""
                
                try:
                    if isinstance(audio, tuple) and len(audio) == 2:
                        sample_rate, audio_data = audio
                        if audio_data is not None and len(audio_data) > 0:
                            user_text, ai_response, audio_path = self.process_audio_input(audio)
                            if user_text and ai_response:
                                history.append({"role": "user", "content": user_text})
                                history.append({"role": "assistant", "content": ai_response})
                                
                                translation = ""
                                words_info = ""
                                
                                if show_trans:
                                    translation = self.translator.translate_to_japanese(ai_response)
                                
                                if show_word:
                                    difficult_words = self.translator.extract_difficult_words(ai_response)
                                    if difficult_words:
                                        word_translations = self.translator.get_word_translations(difficult_words)
                                        words_info = "\n".join([
                                            f"• {w['word']} ({w['pos']}): {word_translations.get(w['word'], 'N/A')}"
                                            for w in difficult_words[:10]
                                        ])
                                    else:
                                        words_info = "難しい単語は見つかりませんでした。"
                                
                                return history, audio_path, translation, words_info
                except Exception as e:
                    print(f"音声自動処理エラー: {e}")
                
                return history, None, "", ""
            
            # 音声入力の変更イベント（録音完了時に自動処理）
            audio_input.change(
                on_audio_change,
                inputs=[audio_input, chatbot, show_translation, show_words],
                outputs=[chatbot, audio_output, translation_box, words_box]
            )
            
            # チェックボックスで表示切り替えと翻訳・単語の取得
            def update_translation(show, history):
                """翻訳表示の切り替えと、最新のAI応答の翻訳を取得"""
                if not show:
                    return gr.update(visible=False, value="")
                
                # 最新のAI応答を取得
                if history and len(history) > 0:
                    # 最新のassistantメッセージを探す
                    for msg in reversed(history):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            translation = self.translator.translate_to_japanese(msg.get("content", ""))
                            return gr.update(visible=True, value=translation)
                
                return gr.update(visible=True, value="翻訳する会話がありません。")
            
            def update_words(show, history):
                """単語表示の切り替えと、最新のAI応答の単語を取得"""
                if not show:
                    return gr.update(visible=False, value="")
                
                # 最新のAI応答を取得
                if history and len(history) > 0:
                    # 最新のassistantメッセージを探す
                    for msg in reversed(history):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            ai_response = msg.get("content", "")
                            difficult_words = self.translator.extract_difficult_words(ai_response)
                            if difficult_words:
                                word_translations = self.translator.get_word_translations(difficult_words)
                                words_info = "\n".join([
                                    f"• {w['word']} ({w['pos']}): {word_translations.get(w['word'], 'N/A')}"
                                    for w in difficult_words[:10]
                                ])
                                return gr.update(visible=True, value=words_info)
                            else:
                                return gr.update(visible=True, value="難しい単語は見つかりませんでした。")
                
                return gr.update(visible=True, value="単語を抽出する会話がありません。")
            
            show_translation.change(
                update_translation,
                inputs=[show_translation, chatbot],
                outputs=[translation_box]
            )
            
            show_words.change(
                update_words,
                inputs=[show_words, chatbot],
                outputs=[words_box]
            )
            
            # 会話履歴のクリアボタン
            def clear_conversation():
                self.conversation_history = []
                return []
            
            clear_btn = gr.Button("会話履歴をクリア", variant="secondary")
            clear_btn.click(
                clear_conversation,
                outputs=[chatbot]
            )
        
        return demo

def main():
    """
    メイン関数
    """
    app = EnglishConversationApp()
    demo = app.create_ui()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())

if __name__ == "__main__":
    main()

