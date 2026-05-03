from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy import signal

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
    
    def transcribe(self, audio_data, sample_rate=16000, debug=False):
        """
        音声データをテキストに変換
        
        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート（デフォルト: 16000Hz）
            debug: デバッグ情報を表示するかどうか
        
        Returns:
            認識されたテキスト
        """
        # Whisperは16000Hzを期待するため、明示的にリサンプリング
        target_sample_rate = 16000
        
        if sample_rate != target_sample_rate:
            print(f"\n🔄 リサンプリング: {sample_rate}Hz → {target_sample_rate}Hz")
            # scipy.signal.resampleを使用してリサンプリング
            num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            original_sample_rate = sample_rate
            sample_rate = target_sample_rate
            print(f"   リサンプリング完了: {len(audio_data)} サンプル")
        else:
            original_sample_rate = sample_rate
        
        # 音声データの統計情報を計算
        audio_duration = len(audio_data) / sample_rate
        audio_rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
        audio_peak = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"🎤 音声認識デバッグ情報")
        print(f"{'='*60}")
        print(f"音声データ長: {len(audio_data)} サンプル ({audio_duration:.2f}秒)")
        print(f"サンプリングレート: {sample_rate} Hz (元: {original_sample_rate} Hz)")
        print(f"RMS (平均音量): {audio_rms:.4f}")
        print(f"ピーク音量: {audio_peak:.4f}")
        print(f"音量レベル (dB): {20 * np.log10(audio_peak + 1e-10):.2f} dB")
        
        # 音声データの長さをチェック（最低0.5秒必要）
        if len(audio_data) < sample_rate * 0.5:
            print(f"⚠️ 警告: 音声データが短すぎます ({audio_duration:.2f}秒)")
            return ""
        
        # VADパラメータを調整（より多くの音声を検出）
        # まずVAD無効で試行（全文を認識）
        print(f"\n📊 VAD設定:")
        print(f"  - 初回試行: VAD無効（全文認識を試みる）")
        
        # VAD無効で試行（全文を認識）
        # initial_promptを改善（より具体的なプロンプトで精度向上）
        initial_prompt = (
            "Hello, my name is Surira. "
            "This is a conversation in English. "
            "The speaker is introducing themselves or having a casual conversation."
        )
        
        segments_no_vad, info_no_vad = self.model.transcribe(
            audio_data, 
            language="en",
            vad_filter=False,  # VADを無効化して全文を認識
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            temperature=0.0,  # 温度を0にしてより確定的な結果を得る
            compression_ratio_threshold=2.4,  # 圧縮率の閾値（繰り返しを検出）
            log_prob_threshold=-1.0,  # ログ確率の閾値（低品質な結果をフィルタ）
            no_speech_threshold=0.6  # 無音検出の閾値
        )
        
        segments_list_no_vad = list(segments_no_vad)
        text_no_vad = " ".join([s.text.strip() for s in segments_list_no_vad if s.text.strip()])
        
        print(f"  - VAD無効時のセグメント数: {len(segments_list_no_vad)}")
        print(f"  - VAD無効時の認識テキスト: '{text_no_vad}'")
        
        # VAD有効で試行（ノイズ除去）
        print(f"\n  - 2回目試行: VAD有効（ノイズ除去）")
        print(f"    - 最小無音時間: 300ms")
        print(f"    - VAD閾値: 0.5")
        
        segments, info = self.model.transcribe(
            audio_data, 
            language="en",
            vad_filter=True,  # 音声検出フィルタを有効化
            vad_parameters=dict(
                min_silence_duration_ms=300,  # 無音検出の閾値
                threshold=0.5  # VADの閾値（デフォルト値）
            ),
            beam_size=5,  # ビームサイズを設定（精度向上）
            best_of=5,  # ベストオブサンプリング（精度向上）
            condition_on_previous_text=False,  # 前のテキストに依存しない（高速化）
            initial_prompt=initial_prompt,  # 改善されたプロンプトを使用
            word_timestamps=True,  # 単語タイムスタンプを有効化（デバッグ用）
            temperature=0.0,  # 温度を0にしてより確定的な結果を得る
            compression_ratio_threshold=2.4,  # 圧縮率の閾値
            log_prob_threshold=-1.0,  # ログ確率の閾値
            no_speech_threshold=0.6  # 無音検出の閾値
        )
        
        # VAD有効の結果も取得
        segments_list_vad = list(segments)
        text_vad = " ".join([s.text.strip() for s in segments_list_vad if s.text.strip()])
        
        print(f"  - VAD有効時のセグメント数: {len(segments_list_vad)}")
        print(f"  - VAD有効時の認識テキスト: '{text_vad}'")
        
        # VAD無効の結果がより良い場合はそれを使用
        # より長いテキストを認識している方を選択
        if len(text_no_vad) > len(text_vad) * 1.2:  # VAD無効の方が20%以上長い
            print(f"\n✅ VAD無効の結果を使用（より多くのテキストを認識: {len(text_no_vad)}文字 vs {len(text_vad)}文字）")
            segments = segments_no_vad
            info = info_no_vad
            segments_list = segments_list_no_vad
        else:
            print(f"\n✅ VAD有効の結果を使用（ノイズ除去済み: {len(text_vad)}文字 vs {len(text_no_vad)}文字）")
            segments_list = segments_list_vad
        
        # 言語情報を表示
        print(f"\n🌐 言語情報:")
        print(f"  - 検出言語: {info.language} (確率: {info.language_probability:.2%})")
        
        # セグメント情報を詳細に表示
        print(f"\n📝 検出されたセグメント数: {len(segments_list)}")
        
        if len(segments_list) == 0:
            print("⚠️ 警告: セグメントが検出されませんでした")
            print("   考えられる原因:")
            print("   - VADが音声を検出できていない")
            print("   - 音声が短すぎる、または音量が低すぎる")
            print("   - ノイズが多すぎる")
            return ""
        
        # 各セグメントの詳細情報を表示
        text_parts = []
        total_duration = 0
        
        print(f"\n{'─'*60}")
        print(f"{'開始時間':<12} {'終了時間':<12} {'信頼度':<10} {'テキスト'}")
        print(f"{'─'*60}")
        
        for i, segment in enumerate(segments_list):
            start_time = segment.start
            end_time = segment.end
            duration = end_time - start_time
            text = segment.text.strip()
            
            # 信頼度スコア（no_speech_probが低いほど信頼度が高い）
            no_speech_prob = getattr(segment, 'no_speech_prob', None)
            confidence = (1.0 - no_speech_prob) * 100 if no_speech_prob is not None else None
            
            if text:
                text_parts.append(text)
                total_duration += duration
                
                confidence_str = f"{confidence:.1f}%" if confidence is not None else "N/A"
                print(f"{start_time:>8.2f}s  {end_time:>8.2f}s  {confidence_str:>8}  {text}")
        
        print(f"{'─'*60}")
        
        # 認識されたテキストの統計
        text = " ".join(text_parts)
        recognized_duration = total_duration
        coverage_ratio = recognized_duration / audio_duration if audio_duration > 0 else 0
        
        print(f"\n📈 認識結果の統計:")
        print(f"  - 認識されたテキスト: '{text}'")
        print(f"  - 認識された音声時間: {recognized_duration:.2f}秒")
        print(f"  - 元の音声時間: {audio_duration:.2f}秒")
        print(f"  - カバレッジ率: {coverage_ratio:.1%}")
        
        if coverage_ratio < 0.5:
            print(f"  ⚠️ 警告: カバレッジ率が低いです（{coverage_ratio:.1%}）")
            print(f"     音声の一部しか認識されていない可能性があります")
            print(f"     考えられる原因:")
            print(f"     - VADが音声の一部を無音として検出している")
            print(f"     - 音声が途切れている、またはノイズが多い")
        
        # 平均信頼度を計算
        if segments_list:
            confidences = [getattr(s, 'no_speech_prob', None) for s in segments_list]
            confidences = [(1.0 - c) * 100 for c in confidences if c is not None]
            if confidences:
                avg_confidence = np.mean(confidences)
                print(f"  - 平均信頼度: {avg_confidence:.1f}%")
        
        print(f"{'='*60}\n")
        
        if not text:
            print("⚠️ 警告: 音声からテキストを抽出できませんでした")
        
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

