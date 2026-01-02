from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import soundfile as sf

class SpeechRecognizer:
    def __init__(self, model_size="small"):
        """
        音声認識モデルの初期化
        
        Args:
            model_size: Whisperモデルのサイズ ("tiny", "base", "small", "medium", "large")
                       デフォルトを"small"に変更（精度向上）
        """
        # GPU使用、compute_type="float16"で高速化
        # "small"モデルは"base"より精度が高いが、速度は少し遅い
        print(f"Loading Whisper model: {model_size} (精度優先)")
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    def transcribe(self, audio_data, sample_rate=16000):
        """
        音声データをテキストに変換
        
        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート（デフォルト: 16000Hz）
        
        Returns:
            認識されたテキスト
        """
        # 音声データをテキストに変換
        # Faster-Whisperは自動的にリサンプリングを行う
        segments, info = self.model.transcribe(
            audio_data, 
            language="en",
            vad_filter=True,  # 音声検出フィルタを有効化
            vad_parameters=dict(
                min_silence_duration_ms=300,  # 500→300に短縮（より敏感に）
                threshold=0.5  # VADの閾値（デフォルトより低く設定）
            ),
            beam_size=5,  # ビームサイズを設定（精度向上）
            best_of=5,  # ベストオブサンプリング（精度向上）
            condition_on_previous_text=False  # 前のテキストに依存しない（高速化）
        )
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    
    def record_audio(self, duration=5, sample_rate=16000):
        """
        マイクから音声を録音
        
        Args:
            duration: 録音時間（秒）
            sample_rate: サンプリングレート
        
        Returns:
            録音された音声データ（numpy配列）
        """
        print(f"録音中... ({duration}秒)")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        return audio.flatten()

