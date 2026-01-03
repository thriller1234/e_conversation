import os
import platform
import subprocess
import tempfile
import json

class TextToSpeech:
    def __init__(self, voice=None):
        """
        音声合成モデルの初期化（PowerShell TTS - 完全オフライン対応）
        
        Args:
            voice: 使用する音声名（Noneの場合は英語音声を自動選択）
        """
        print("Loading TTS model (PowerShell TTS - 完全オフライン対応)...")
        self.preferred_voice_name = voice  # ユーザー指定の音声名
        self.detected_voice_name = None  # 検出された音声名
        
        if platform.system() == 'Windows':
            try:
                # PowerShellで利用可能な音声を取得（SAPI5）
                ps_script = r'''
Add-Type -AssemblyName System.Speech
$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synthesizer.GetInstalledVoices() | ForEach-Object {
    $voice = $_.VoiceInfo
    [PSCustomObject]@{
        Name = $voice.Name
        Description = $voice.Description
        Culture = $voice.Culture.Name
    } | ConvertTo-Json -Compress
}
'''
                result = subprocess.run(
                    ['powershell', '-Command', ps_script],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    voices_info = []
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            try:
                                voices_info.append(json.loads(line))
                            except:
                                pass
                    
                    # 英語音声を検索
                    english_voices = [
                        v for v in voices_info
                        if 'en-US' in v.get('Culture', '') or 'en-GB' in v.get('Culture', '')
                    ]
                    
                    # ユーザー指定の音声名がある場合は、それを優先
                    if self.preferred_voice_name:
                        matching_voice = None
                        for v in english_voices:
                            desc = v.get('Description', '')
                            name = v.get('Name', '')
                            if (self.preferred_voice_name.lower() in desc.lower() or 
                                self.preferred_voice_name.lower() in name.lower()):
                                matching_voice = v
                                break
                        
                        if matching_voice:
                            self.detected_voice_name = matching_voice.get('Description', matching_voice.get('Name', ''))
                            print(f"✅ 指定された音声を使用: {self.detected_voice_name}")
                        else:
                            print(f"⚠️ 指定された音声 '{self.preferred_voice_name}' が見つかりません。自動検出を試みます...")
                            self.preferred_voice_name = None
                    
                    # 自動検出（ユーザー指定がない場合、または指定された音声が見つからない場合）
                    if not self.detected_voice_name and english_voices:
                        selected_voice = english_voices[0]
                        self.detected_voice_name = selected_voice.get('Description', selected_voice.get('Name', ''))
                        print(f"✅ 英語音声を自動選択: {self.detected_voice_name}")
                    
                    # 最終的な音声名を設定
                    if self.detected_voice_name:
                        self.preferred_voice_name = self.detected_voice_name
                else:
                    print(f"⚠️ PowerShellで音声リスト取得エラー: {result.stderr}")
            except Exception as e:
                print(f"⚠️ PowerShell音声リスト取得エラー: {e}")
        
        print("TTS model loaded successfully!")
    
    def speak(self, text, output_path="output.wav"):
        """
        テキストを音声ファイルに変換（PowerShell TTSを使用）
        
        Args:
            text: 音声化するテキスト
            output_path: 出力ファイルパス
        
        Returns:
            生成された音声ファイルのパス（エラー時はNone）
        """
        if platform.system() != 'Windows':
            print("⚠️ PowerShell TTSはWindows環境でのみ利用可能です")
            return None
        
        try:
            # 絶対パスに変換
            output_path = os.path.abspath(output_path)
            
            # 出力ディレクトリが存在することを確認
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 既存のファイルを確実に削除（リトライ付き）
            max_retries = 5
            for retry in range(max_retries):
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                        import time
                        time.sleep(0.1 * (retry + 1))
                        if not os.path.exists(output_path):
                            break
                    except Exception as e:
                        if retry == max_retries - 1:
                            print(f"⚠️ ファイル削除に失敗しました（{max_retries}回試行）: {e}")
                            # 別のファイル名を試す
                            base_name = os.path.splitext(output_path)[0]
                            output_path = f"{base_name}_{retry}.wav"
                        else:
                            import time
                            time.sleep(0.1 * (retry + 1))
                else:
                    break
            
            # 一時的なPowerShellスクリプトファイルを作成
            temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8')
            try:
                # PowerShellスクリプトを書き込み
                # 高品質音声（Neural/Natural/Aria）を優先的に使用
                # 初期化時に選択された音声名を使用（指定されている場合）
                preferred_voice = self.preferred_voice_name if self.preferred_voice_name else ""
                script_content = f'''Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer

# 高品質音声（Neural/Natural/Aria）を検索して設定
$voices = $speak.GetInstalledVoices()
$highQualityVoice = $null
$standardVoice = $null

# 初期化時に選択された音声名が指定されている場合は、それを優先
$preferredVoiceName = "{preferred_voice}"
if ($preferredVoiceName -ne "") {{
    # まず、完全一致を試す
    foreach ($voice in $voices) {{
        $voiceInfo = $voice.VoiceInfo
        $voiceDesc = $voiceInfo.Description
        $voiceName = $voiceInfo.Name
        # 完全一致を優先
        if (($voiceDesc -eq $preferredVoiceName -or $voiceName -eq $preferredVoiceName) -and 
            $voiceInfo.Culture.Name -like "en-*") {{
            $highQualityVoice = $voiceInfo
            break
        }}
    }}
    
    # 完全一致が見つからない場合、部分一致を試す
    if ($highQualityVoice -eq $null) {{
        foreach ($voice in $voices) {{
            $voiceInfo = $voice.VoiceInfo
            $voiceDesc = $voiceInfo.Description
            $voiceName = $voiceInfo.Name
            # 部分一致（大文字小文字を区別しない）
            if (($voiceDesc -like "*$preferredVoiceName*" -or $voiceName -like "*$preferredVoiceName*") -and 
                $voiceInfo.Culture.Name -like "en-*") {{
                $highQualityVoice = $voiceInfo
                break
            }}
        }}
    }}
    
}}

# 指定された音声が見つからない場合、英語音声を検索
if ($highQualityVoice -eq $null) {{
    foreach ($voice in $voices) {{
        $voiceInfo = $voice.VoiceInfo
        if ($voiceInfo.Culture.Name -like "en-*") {{
            $highQualityVoice = $voiceInfo
            break
        }}
    }}
}}

# 音声を設定
if ($highQualityVoice -ne $null) {{
    # Descriptionを優先的に使用（なければNameを使用）
    $voiceToUse = $highQualityVoice.Description
    if ($voiceToUse -eq $null -or $voiceToUse -eq "") {{
        $voiceToUse = $highQualityVoice.Name
    }}
    try {{
        $speak.SelectVoice($voiceToUse)
    }} catch {{
        # Descriptionで失敗した場合、Nameを試す
        try {{
            $speak.SelectVoice($highQualityVoice.Name)
        }} catch {{
            Write-Host "音声選択エラー: $_"
        }}
    }}
}}

# 出力設定（WAVファイル、高品質）
$speak.SetOutputToWaveFile([string]"{output_path}")
$speak.Rate = -1  # 速度を少し上げる
$speak.Volume = 100  # 音量を最大に

# 音声を生成
$speak.Speak([string]@"
{text}
"@)
$speak.Dispose()
'''
                temp_script.write(script_content)
                temp_script.close()
                
                # PowerShellで実行（-ExecutionPolicy Bypassでスクリプト実行を許可）
                result = subprocess.run(
                    ['powershell', '-ExecutionPolicy', 'Bypass', '-File', temp_script.name],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=os.path.dirname(output_path)
                )
                
                # 一時スクリプトを削除
                try:
                    os.remove(temp_script.name)
                except:
                    pass
                
                if result.returncode == 0:
                    # ファイルが生成されるまで待機（最大2秒）
                    import time
                    max_wait = 2.0
                    wait_interval = 0.1
                    waited = 0.0
                    while waited < max_wait:
                        if os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            if file_size > 0:
                                print(f"✅ 音声ファイルを生成しました（PowerShell）: {output_path} ({file_size} bytes)")
                                return output_path
                        time.sleep(wait_interval)
                        waited += wait_interval
                    
                    # タイムアウトした場合
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        if file_size == 0:
                            print(f"⚠️ 音声ファイルが空です: {output_path}")
                            return None
                    else:
                        print(f"⚠️ 音声ファイルが生成されませんでした: {output_path}")
                        if result.stderr:
                            print(f"   PowerShellエラー: {result.stderr}")
                        return None
                else:
                    print(f"⚠️ PowerShell TTSエラー（リターンコード: {result.returncode}）")
                    if result.stderr:
                        print(f"   エラー内容: {result.stderr}")
                    if result.stdout:
                        print(f"   出力: {result.stdout}")
                    return None
            finally:
                # 一時スクリプトを確実に削除
                try:
                    if os.path.exists(temp_script.name):
                        os.remove(temp_script.name)
                except:
                    pass
        except Exception as e:
            print(f"⚠️ PowerShell TTSエラー: {e}")
            import traceback
            print(f"トレースバック: {traceback.format_exc()}")
            return None
    
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
        result = self.speak(text, temp_path)
        
        if result is None:
            # エラー時は空の配列を返す
            import numpy as np
            return np.array([]), sample_rate
        
        import soundfile as sf
        audio_data, sr = sf.read(temp_path)
        
        # 一時ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return audio_data, sr
    
    @staticmethod
    def list_voices():
        """
        利用可能な音声のリストを取得（PowerShell経由）
        
        Returns:
            音声のリスト（辞書のリスト）
        """
        if platform.system() != 'Windows':
            return []
        
        try:
            ps_script = '''
Add-Type -AssemblyName System.Speech
$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synthesizer.GetInstalledVoices() | ForEach-Object {
    $voice = $_.VoiceInfo
    [PSCustomObject]@{
        Id = $voice.Id
        Name = $voice.Description
        Culture = $voice.Culture.Name
        IsHighQuality = (($voice.Description -like "*Neural*") -or ($voice.Description -like "*Natural*"))
    } | ConvertTo-Json -Compress
}
'''
            result = subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                voice_list = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            voice_info = json.loads(line)
                            voice_list.append({
                                'id': voice_info.get('Id', ''),
                                'name': voice_info.get('Name', ''),
                                'culture': voice_info.get('Culture', '')
                            })
                        except:
                            pass
                return voice_list
            else:
                return []
        except Exception as e:
            print(f"⚠️ 音声リスト取得エラー: {e}")
            return []
