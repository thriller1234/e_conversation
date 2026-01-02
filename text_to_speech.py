import os
import asyncio
import edge_tts

class TextToSpeech:
    def __init__(self, voice="en-US-AriaNeural"):
        """
        音声合成モデルの初期化
        
        Args:
            voice: 使用する音声（デフォルト: en-US-AriaNeural）
                  利用可能な音声: en-US-AriaNeural, en-US-JennyNeural, en-GB-SoniaNeural など
        """
        print("Loading TTS model (edge-tts)...")
        self.voice = voice
        print(f"TTS model loaded successfully! Voice: {voice}")
    
    def speak(self, text, output_path="output.wav"):
        """
        テキストを音声ファイルに変換
        
        Args:
            text: 音声化するテキスト
            output_path: 出力ファイルパス
        
        Returns:
            生成された音声ファイルのパス
        """
        async def generate_speech():
            communicate = edge_tts.Communicate(text=text, voice=self.voice)
            await communicate.save(output_path)
        
        # 非同期関数を実行
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(generate_speech())
        return output_path
    
    def speak_to_array(self, text, sample_rate=22050):
        """
        テキストを音声配列に変換（リアルタイム再生用）
        
        Args:
            text: 音声化するテキスト
            sample_rate: サンプリングレート
        
        Returns:
            音声データ（numpy配列）、サンプリングレート
        """
        # 一時ファイルに保存してから読み込む
        temp_path = "temp_tts_output.wav"
        self.speak(text, temp_path)
        
        import soundfile as sf
        audio_data, sr = sf.read(temp_path)
        
        # 一時ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return audio_data, sr
    
    @staticmethod
    def list_voices():
        """
        利用可能な音声のリストを取得
        
        Returns:
            音声のリスト
        """
        async def get_voices():
            voices = await edge_tts.list_voices()
            return voices
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(get_voices())
