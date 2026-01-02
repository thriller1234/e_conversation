from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# bitsandbytesの利用可能性を確認
try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

# Hugging Faceのタイムアウト設定を大幅に増やす（低速回線対応）
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"  # 60分に設定
os.environ["HF_HUB_DOWNLOAD_CHUNK_SIZE"] = "1048576"  # 1MBチャンク（デフォルトより小さい）

class LLMHandler:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        """
        LLMハンドラーの初期化
        
        Args:
            model_name: 使用するLLMモデル名
        """
        print(f"Loading LLM model: {model_name}...")
        print("⚠️ 低速回線対応モード: タイムアウト60分、リトライ機能有効")
        
        # トークナイザーの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # モデル読み込みの設定（低速回線対応）
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        # snapshot_downloadの設定を環境変数で制御
        # 並列ダウンロードを無効化（順次ダウンロード）
        os.environ["HF_HUB_DOWNLOAD_MAX_WORKERS"] = "1"
        
        # bitsandbytesが利用可能な場合のみ8bit量子化を使用
        if BITSANDBYTES_AVAILABLE:
            try:
                model_kwargs["load_in_8bit"] = True
                print("Using 8-bit quantization (bitsandbytes)")
            except Exception as e:
                print(f"Warning: 8-bit quantization failed: {e}")
                print("Falling back to float16")
        else:
            print("bitsandbytes not available, using float16 (requires more VRAM)")
        
        # モデルの読み込み（リトライ機能付き）
        max_retries = 10  # 最大10回リトライ（低速回線対応）
        retry_delay = 15  # リトライ間隔（秒）
        
        # 初回実行時に不完全なファイルをクリーンアップ
        self._clear_incomplete_files(model_name)
        
        for attempt in range(max_retries):
            try:
                print(f"\n📥 モデルダウンロード開始（試行 {attempt + 1}/{max_retries}）...")
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                print("✅ モデルのダウンロードが完了しました！")
                
                # 推論モードに設定（高速化）
                self.model.eval()
                
                # torch.compile()でモデルをコンパイル（PyTorch 2.0+で高速化）
                try:
                    if hasattr(torch, 'compile'):
                        print("🚀 torch.compile()でモデルを最適化中...")
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        print("✅ モデルの最適化が完了しました！")
                except Exception as e:
                    print(f"⚠️ torch.compile()の最適化をスキップ: {e}")
                
                # cuDNNの最適化を有効化
                torch.backends.cudnn.benchmark = True
                
                break
            except (ConnectionResetError, TimeoutError, OSError, Exception) as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                is_retryable = (
                    "ConnectionResetError" in error_msg or
                    "Read timed out" in error_msg or
                    "Connection broken" in error_msg or
                    "does not appear to have a file" in error_msg or
                    "ChunkedEncodingError" in error_type or
                    error_type == "OSError"
                )
                
                if is_retryable and attempt < max_retries - 1:
                    print(f"\n⚠️ ダウンロードエラーが発生しました: {error_type}")
                    print(f"   エラー内容: {error_msg[:200]}...")
                    print(f"⏳ {retry_delay}秒後に再試行します（残り試行回数: {max_retries - attempt - 1}）...")
                    import time
                    time.sleep(retry_delay)
                    
                    # 不完全なファイルを削除
                    print("🧹 不完全なファイルをクリーンアップ中...")
                    self._clear_incomplete_files(model_name)
                else:
                    print(f"\n❌ ダウンロードに失敗しました（最大リトライ回数に達しました）")
                    print(f"   最後のエラー: {error_type}: {error_msg[:200]}")
                    print("\n💡 ヒント:")
                    print("   - ネットワーク接続を確認してください")
                    print("   - キャッシュをクリアして再試行: python clear_hf_cache.py")
                    raise
        
        # 英会話教師としてのシステムプロンプト
        self.system_prompt = """You are a friendly English conversation teacher. 
Have natural conversations with the student to help them improve their English.
Use clear, natural English at an intermediate level.
Be encouraging and patient."""
        print("LLM model loaded successfully!")
    
    def _clear_incomplete_files(self, model_name):
        """
        不完全なダウンロードファイルを削除（低速回線対応）
        
        Args:
            model_name: モデル名
        """
        from pathlib import Path
        
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir_name = f"models--{model_name.replace('/', '--')}"
        model_cache_path = cache_dir / model_dir_name
        
        if not model_cache_path.exists():
            return
        
        cleared_count = 0
        
        # .incompleteファイルを削除
        for incomplete_file in model_cache_path.rglob("*.incomplete"):
            try:
                incomplete_file.unlink()
                cleared_count += 1
            except Exception:
                pass
        
        # .safetensorsファイルのサイズをチェック
        # 期待されるファイルサイズ（Llama 3.1 8Bの場合）
        expected_sizes = {
            "model-00001-of-00004.safetensors": 4.98 * 1024 * 1024 * 1024,  # 約4.98GB
            "model-00002-of-00004.safetensors": 5.00 * 1024 * 1024 * 1024,  # 約5.00GB
            "model-00003-of-00004.safetensors": 4.92 * 1024 * 1024 * 1024,  # 約4.92GB
            "model-00004-of-00004.safetensors": 1.17 * 1024 * 1024 * 1024,  # 約1.17GB
        }
        
        for safetensors_file in model_cache_path.rglob("*.safetensors"):
            try:
                file_size = safetensors_file.stat().st_size
                file_name = safetensors_file.name
                
                # 期待されるサイズと比較（10%の誤差を許容）
                if file_name in expected_sizes:
                    expected_size = expected_sizes[file_name]
                    if file_size < expected_size * 0.9:  # 90%未満なら不完全とみなす
                        print(f"   🗑️  不完全なファイルを削除: {file_name} ({file_size / (1024**3):.2f}GB / 期待値: {expected_size / (1024**3):.2f}GB)")
                        safetensors_file.unlink()
                        cleared_count += 1
                else:
                    # 期待されるサイズが不明な場合、500MB未満なら削除
                    if file_size < 500 * 1024 * 1024:
                        print(f"   🗑️  小さすぎるファイルを削除: {file_name} ({file_size / (1024**3):.2f}GB)")
                        safetensors_file.unlink()
                        cleared_count += 1
            except Exception as e:
                # ファイルがロックされている場合などはスキップ
                pass
        
        if cleared_count > 0:
            print(f"   ✅ {cleared_count}個の不完全なファイルを削除しました")
        else:
            print("   ℹ️ 削除する不完全なファイルはありませんでした")
    
    def generate_response(self, user_input, conversation_history=[]):
        """
        ユーザー入力に対する応答を生成
        
        Args:
            user_input: ユーザーの入力テキスト
            conversation_history: 会話履歴（オプション）
        
        Returns:
            AIの応答テキスト
        """
        # 会話履歴を含めてプロンプト作成
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # トークン化して生成
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        
        # 推論パラメータの最適化（高速化）
        with torch.no_grad():  # 勾配計算を無効化（メモリ節約と高速化）
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=120,  # 150→120に削減（高速化）
                temperature=0.7,
                top_p=0.9,  # 核サンプリング（品質と速度のバランス）
                top_k=50,  # top-kサンプリング（高速化）
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # KVキャッシュを有効化（高速化）
                num_beams=1,  # ビームサーチを無効化（高速化）
                repetition_penalty=1.1  # 繰り返しを抑制
            )
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def translate_to_japanese(self, english_text):
        """
        LLMを使用して英語を日本語に翻訳（高速化版）
        
        Args:
            english_text: 翻訳する英語テキスト
        
        Returns:
            日本語訳
        """
        # プロンプトを簡潔に（高速化）
        prompt = f"日本語に翻訳: {english_text}"
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        
        # 推論パラメータの最適化（翻訳は短いのでmax_new_tokensを削減）
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=60,  # 100→60に削減（高速化）
                temperature=0.2,  # 0.3→0.2に削減（より確定的で高速）
                top_p=0.9,
                top_k=30,  # top-kを小さく（高速化）
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # KVキャッシュを有効化
                num_beams=1,  # ビームサーチを無効化（高速化）
                repetition_penalty=1.05
            )
        translation = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return translation.strip()

