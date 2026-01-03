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
        self.recognizer = SpeechRecognizer(model_size="small")  # 精度向上のためsmallに変更
        self.llm = LLMHandler()
        # 音声名を直接指定する場合（Noneの場合は自動検出）
        self.tts = TextToSpeech(voice=None)
        self.translator = TranslationHelper(self.llm)
        self.conversation_history = []
        self.is_recording = False
        self.audio_buffer = []
        self.temp_audio_files = []  # 一時音声ファイルのリスト（終了時に削除）
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
        
        # 音声データの前処理（無音部分のトリム、ノイズ除去、音量正規化）
        audio_data = self._preprocess_audio(audio_data, sample_rate)
        
        # 音声認識（デバッグモード有効）
        try:
            print("DEBUG: 音声認識を開始...")
            user_text = self.recognizer.transcribe(audio_data, sample_rate=sample_rate, debug=True)
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
        
        # 音声合成（エラー時も会話を続行）
        audio_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_path = self.tts.speak(ai_response, temp_file.name)
            if audio_path is None:
                print("⚠️ 音声合成に失敗しましたが、テキスト会話は続行します")
            else:
                # 一時ファイルのパスを記録（終了時に削除）
                self.temp_audio_files.append(audio_path)
        except Exception as e:
            import traceback
            print(f"⚠️ 音声合成エラー: {e}")
            print(f"トレースバック: {traceback.format_exc()}")
            print("⚠️ テキスト会話は続行します")
        
        return user_text, ai_response, audio_path
    
    def _preprocess_audio(self, audio_data, sample_rate):
        """
        音声データの前処理（無音部分のトリム、ノイズ除去、音量正規化）
        
        Args:
            audio_data: 音声データ（numpy配列、float32）
            sample_rate: サンプリングレート
        
        Returns:
            処理済みの音声データ
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            return audio_data
        
        # 1. 無音部分のトリム（前後の無音を削除）
        # 音量の閾値（-40dB相当、約0.01）
        silence_threshold = 0.01
        
        # 音声の絶対値を計算
        abs_audio = np.abs(audio_data)
        
        # 前後の無音部分を検出
        # 連続する無音の最小長さ（0.1秒）
        min_silence_length = int(sample_rate * 0.1)
        
        # 前の無音部分を検出
        start_idx = 0
        for i in range(len(abs_audio) - min_silence_length):
            if np.max(abs_audio[i:i+min_silence_length]) > silence_threshold:
                start_idx = i
                break
        
        # 後の無音部分を検出
        end_idx = len(abs_audio)
        for i in range(len(abs_audio) - min_silence_length, 0, -1):
            if np.max(abs_audio[i:i+min_silence_length]) > silence_threshold:
                end_idx = i + min_silence_length
                break
        
        # トリム実行（最低0.1秒は残す）
        min_length = int(sample_rate * 0.1)
        if end_idx - start_idx < min_length:
            # 音声が短すぎる場合は中央部分を保持
            center = len(audio_data) // 2
            start_idx = max(0, center - min_length // 2)
            end_idx = min(len(audio_data), center + min_length // 2)
        
        trimmed_audio = audio_data[start_idx:end_idx]
        
        if trimmed_audio.size == 0:
            print("警告: トリム後の音声データが空です")
            return audio_data
        
        print(f"DEBUG: 音声トリム: {len(audio_data)} → {len(trimmed_audio)} サンプル ({start_idx/sample_rate:.2f}s - {end_idx/sample_rate:.2f}s)")
        
        # 2. 音量正規化（ピークが-3dBになるように調整）
        max_amplitude = np.max(np.abs(trimmed_audio))
        if max_amplitude > 0:
            # -3dB = 10^(-3/20) ≈ 0.708
            target_peak = 0.708
            if max_amplitude < target_peak:
                # 音量が小さい場合は正規化
                normalized_audio = trimmed_audio * (target_peak / max_amplitude)
                # クリッピングを防ぐ（最大1.0）
                normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                print(f"DEBUG: 音量正規化: ピーク {max_amplitude:.4f} → {np.max(np.abs(normalized_audio)):.4f}")
                return normalized_audio
        
        return trimmed_audio
    
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
        
        # 音声合成（エラー時も会話を続行）
        audio_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_path = self.tts.speak(ai_response, temp_file.name)
            if audio_path is None:
                print("⚠️ 音声合成に失敗しましたが、テキスト会話は続行します")
            else:
                # 一時ファイルのパスを記録（終了時に削除）
                self.temp_audio_files.append(audio_path)
        except Exception as e:
            import traceback
            print(f"⚠️ 音声合成エラー: {e}")
            print(f"トレースバック: {traceback.format_exc()}")
            print("⚠️ テキスト会話は続行します")
        
        return user_text, ai_response, audio_path
    
    def create_ui(self):
        """
        Gradio UIの作成
        
        Returns:
            Gradio Blocksオブジェクト
        """
        with gr.Blocks() as demo:
            # カスタムJavaScriptでマイク設定を保持
            demo.load(
                fn=None,
                js="""
                function() {
                    // ページ読み込み時にマイク設定を復元
                    setTimeout(() => {
                        const savedDeviceId = localStorage.getItem('gradio_microphone_device');
                        if (savedDeviceId && navigator.mediaDevices) {
                            // マイクデバイスリストを取得して設定を復元
                            navigator.mediaDevices.enumerateDevices().then(devices => {
                                const audioInputs = devices.filter(device => device.kind === 'audioinput');
                                const savedDevice = audioInputs.find(device => device.deviceId === savedDeviceId);
                                if (savedDevice) {
                                    console.log('マイク設定を復元:', savedDevice.label);
                                }
                            });
                        }
                        
                        // マイク使用時にデバイスIDを保存
                        const audioInputs = document.querySelectorAll('input[type="file"][accept*="audio"]');
                        audioInputs.forEach(input => {
                            input.addEventListener('change', function() {
                                navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                                    const tracks = stream.getAudioTracks();
                                    if (tracks.length > 0) {
                                        localStorage.setItem('gradio_microphone_device', tracks[0].getSettings().deviceId);
                                        console.log('マイク設定を保存:', tracks[0].getSettings().deviceId);
                                    }
                                    stream.getTracks().forEach(track => track.stop());
                                });
                            });
                        });
                    }, 1000);
                    return [];
                }
                """
            )
            
            gr.Markdown("# 🎓 AI英会話学習システム")
            gr.Markdown("マイクまたはテキストで英語で会話を始めましょう！")
            
            with gr.Row():
                with gr.Column():
                    # マイク設定を保持するためのカスタムJavaScriptを追加
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
            
            # マイク設定を保存するための状態変数
            microphone_state = gr.State(value=None)
            
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
            
            # 音声入力の自動処理は無効化（送信ボタンでのみ処理）
            # audio_input.changeイベントを削除して、送信ボタンを押した時だけ処理するように変更
            
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
                            ai_response = msg.get("content", "")
                            # 文字列でない場合は文字列に変換
                            if not isinstance(ai_response, str):
                                if isinstance(ai_response, list):
                                    ai_response = " ".join(str(x) for x in ai_response)
                                else:
                                    ai_response = str(ai_response)
                            
                            if not ai_response:
                                continue
                                
                            translation = self.translator.translate_to_japanese(ai_response)
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
                            # 文字列でない場合は文字列に変換
                            if not isinstance(ai_response, str):
                                if isinstance(ai_response, list):
                                    ai_response = " ".join(str(x) for x in ai_response)
                                else:
                                    ai_response = str(ai_response)
                            
                            if not ai_response:
                                continue
                                
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
    
    try:
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
    except KeyboardInterrupt:
        print("\n⚠️ アプリケーションを終了します...")
    finally:
        # 一時音声ファイルを削除
        print("🧹 一時音声ファイルを削除中...")
        deleted_count = 0
        for audio_file in app.temp_audio_files:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    deleted_count += 1
            except Exception as e:
                print(f"⚠️ ファイル削除エラー: {audio_file} - {e}")
        print(f"✅ {deleted_count}個の一時ファイルを削除しました")
        
        # 古い一時ファイルも削除（過去のセッションで残ったファイル）
        try:
            import glob
            temp_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")
            old_files = glob.glob(os.path.join(temp_dir, "tmp*.wav"))
            old_deleted = 0
            for old_file in old_files:
                try:
                    # 24時間以上古いファイルを削除
                    import time
                    file_age = time.time() - os.path.getmtime(old_file)
                    if file_age > 86400:  # 24時間 = 86400秒
                        os.remove(old_file)
                        old_deleted += 1
                except:
                    pass
            if old_deleted > 0:
                print(f"✅ {old_deleted}個の古い一時ファイルを削除しました")
        except Exception as e:
            print(f"⚠️ 古い一時ファイルの削除エラー: {e}")

if __name__ == "__main__":
    main()

