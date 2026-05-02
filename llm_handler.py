from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

try:
    from huggingface_hub.errors import GatedRepoError
except ImportError:
    GatedRepoError = ()  # type: ignore

# プリセット（light=Qwen ゲートなし / balanced・heavy=Meta Llama は HF で利用申請が必要）
_LLM_PRESETS = {
    "light": "Qwen/Qwen2.5-1.5B-Instruct",
    "balanced": "meta-llama/Llama-3.2-3B-Instruct",
    "heavy": "meta-llama/Llama-3.1-8B-Instruct",
}


def _read_llm_local_only() -> bool:
    """
    環境変数 LLM_LOCAL_ONLY。
    True（既定）: キャッシュからのみ読み込み。False: キャッシュを試し、無ければ Hub から取得。
    """
    raw = os.environ.get("LLM_LOCAL_ONLY", "").strip().lower()
    if raw == "":
        return True
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    print(f"[LLM] 不明な LLM_LOCAL_ONLY={raw!r}。true として扱います。")
    return True


def _load_tokenizer(model_name: str, try_remote_on_miss: bool):
    """try_remote_on_miss が False のときはキャッシュのみ。True のときはキャッシュ失敗後に Hub。"""
    if not try_remote_on_miss:
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            raise RuntimeError(
                f"LLM_LOCAL_ONLY=true: キャッシュにトークナイザー ({model_name}) がありません。"
                "初回取得時は LLM_LOCAL_ONLY=false を指定してください。"
            ) from e
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✅ トークナイザーをローカルキャッシュから読み込みました")
        return tok
    except Exception:
        print(
            "⚠️ ローカルキャッシュにトークナイザーが見つかりません。"
            "Hub からの取得を試みます..."
        )
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        except Exception as e2:
            if _is_gated_or_auth_error(e2):
                raise RuntimeError(
                    f"モデル「{model_name}」は Hugging Face でゲート制限されています。"
                    "モデルページで利用申請を承認されたアカウントで `hf auth login` してください。\n"
                    "申請不要で試す場合: LLM_PRESET=light "
                    f"（{_LLM_PRESETS['light']}）"
                ) from e2
            raise


def _is_gated_or_auth_error(exc: BaseException) -> bool:
    if GatedRepoError and isinstance(exc, GatedRepoError):
        return True
    low = str(exc).lower()
    return "gated" in low or ("403" in low and "restricted" in low)


def resolve_llm_model_name(explicit: str | None = None) -> str:
    """
    使用する Hugging Face モデル ID を決定する。

    優先順位:
    1. 引数 explicit（アプリから直接指定した場合）
    2. 環境変数 LLM_PRESET（light / balanced / heavy）
    """
    if explicit and explicit.strip():
        return explicit.strip()
    preset = os.environ.get("LLM_PRESET", "balanced").strip().lower()
    if preset not in _LLM_PRESETS:
        print(
            f"[LLM] 不明な LLM_PRESET={preset!r}。"
            f"有効: {', '.join(_LLM_PRESETS.keys())}。balanced を使います。"
        )
        preset = "balanced"
    resolved = _LLM_PRESETS[preset]
    print(f"[LLM] preset={preset} -> {resolved}")
    return resolved


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
    def __init__(self, model_name: str | None = None):
        """
        LLMハンドラーの初期化

        Args:
            model_name: 使用する Hugging Face モデル ID。None の場合は
                resolve_llm_model_name()（環境変数 LLM_PRESET）で決定。
        """
        model_name = resolve_llm_model_name(model_name)
        print(f"Loading LLM model: {model_name}...")
        print("⚠️ 低速回線対応モード: タイムアウト60分、リトライ機能有効")
        local_only = _read_llm_local_only()
        try_remote_on_miss = not local_only
        if local_only:
            print("[LLM] LLM_LOCAL_ONLY=true（既定）: キャッシュのみ使用。未キャッシュなら LLM_LOCAL_ONLY=false で取得してください。")
        else:
            print("[LLM] LLM_LOCAL_ONLY=false: キャッシュを優先し、無ければ Hub から取得します。")

        self.tokenizer = _load_tokenizer(model_name, try_remote_on_miss)

        # モデル読み込みの設定（低速回線対応）
        # device_map="auto"はCPUオフロードを引き起こす可能性があるため、
        # 明示的にGPUに配置する設定に変更
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "cuda:0"  # 明示的にGPUに配置（CPUオフロードを回避）
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

        print("\n📂 ローカルキャッシュからモデルを読み込み中...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **{**model_kwargs, "local_files_only": True}
            )
            print("✅ モデルをローカルキャッシュから読み込みました！")
            self.model.eval()
            torch.backends.cudnn.benchmark = True
        except Exception as cache_err:
            if local_only:
                raise RuntimeError(
                    f"LLM_LOCAL_ONLY=true: キャッシュにモデル ({model_name}) がありません。"
                    "初回取得時は LLM_LOCAL_ONLY=false を指定してください。"
                ) from cache_err
            print(
                f"⚠️ ローカルキャッシュにモデルが見つかりません: {type(cache_err).__name__}"
            )
            print("📥 Hub からの取得を試みます...")
            for attempt in range(max_retries):
                try:
                    print(f"\n📥 モデルダウンロード開始（試行 {attempt + 1}/{max_retries}）...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, **{**model_kwargs, "local_files_only": False}
                    )
                    print("✅ モデルのダウンロードが完了しました！")
                    
                    # 推論モードに設定（高速化）
                    self.model.eval()
                    
                    # torch.compile()はCPUオフロードと互換性がないため無効化
                    # CPUオフロードが発生している場合（"Some parameters are on the meta device"）は
                    # torch.compile()を使用するとエラーが発生する
                    # 代わりに他の最適化手法を使用
                    print("ℹ️ torch.compile()はCPUオフロードと互換性がないためスキップ")
                    
                    # cuDNNの最適化を有効化
                    torch.backends.cudnn.benchmark = True
                    
                    break
                except (ConnectionResetError, TimeoutError, OSError, Exception) as e:
                    if _is_gated_or_auth_error(e):
                        raise RuntimeError(
                            f"モデル「{model_name}」へのアクセスが拒否されました（ゲート制限または認可）。"
                            "Hugging Face で利用申請を承認し `hf auth login` するか、"
                            "LLM_PRESET=light（Qwen・申請不要）を試してください。"
                        ) from e
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
        
        # 英会話教師としてのシステムプロンプト（訂正なし・会話のみ）
        self.system_prompt_conversation_only = """You are a friendly English conversation teacher.
Have natural conversations with the student to help them improve their English.
Use clear, natural English at an intermediate level.
Be encouraging and patient."""

        # 会話のあいだで学習者の英語を優しく訂正する（既定）
        self.system_prompt_with_correction = """You are a friendly English conversation teacher.
Have natural, spoken-style conversations with the student in English.
Use clear English at an intermediate level. Be encouraging and patient.

When the student's message has clear English mistakes (grammar, word choice, spelling, or unnatural phrasing), help them learn:
- Briefly point out 1-2 of the most important issues (not a long lesson).
- Say what sounds better and why in one short phrase if helpful.
- Then continue the conversation naturally so it still feels like a chat, not an exam.
If their English is already fine for the situation, do not invent mistakes—just reply normally."""
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
    
    def generate_response(
        self,
        user_input,
        conversation_history=None,
        *,
        feedback_corrections: bool = True,
    ):
        """
        ユーザー入力に対する応答を生成
        
        Args:
            user_input: ユーザーの入力テキスト
            conversation_history: 会話履歴（オプション）
            feedback_corrections: True のとき、学習者の英語の誤りを簡潔に訂正してから会話を続ける
        
        Returns:
            AIの応答テキスト
        """
        if conversation_history is None:
            conversation_history = []
        system = (
            self.system_prompt_with_correction
            if feedback_corrections
            else self.system_prompt_conversation_only
        )
        # 会話履歴を含めてプロンプト作成
        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # トークン化して生成
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        
        # attention_maskを明示的に設定（警告回避）
        # pad_token_idがNoneの場合は、すべて1のattention_maskを作成
        if self.tokenizer.pad_token_id is None:
            attention_mask = torch.ones_like(inputs).long().to("cuda")
        else:
            attention_mask = (inputs != self.tokenizer.pad_token_id).long().to("cuda")
        
        # 推論パラメータの最適化（高速化・会話速度向上）
        # 訂正フィードバックがあると応答がやや長くなるため、訂正ありのときだけ上限を少し上げる
        max_new = 110 if feedback_corrections else 80
        with torch.no_grad():  # 勾配計算を無効化（メモリ節約と高速化）
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,  # attention_maskを明示的に設定
                max_new_tokens=max_new,
                temperature=0.5,  # 0.6→0.5に削減（より確定的で高速）
                top_p=0.8,  # 0.85→0.8に削減（高速化）
                top_k=25,  # 30→25に削減（高速化）
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # KVキャッシュを有効化（高速化）
                num_beams=1,  # ビームサーチを無効化（高速化）
                repetition_penalty=1.1,  # 繰り返しを抑制
                # early_stoppingはnum_beams>1の時のみ有効なため削除
            )
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def translate_to_japanese(self, english_text):
        """
        LLMを使用して英語を日本語に翻訳（超高速化版）
        
        Args:
            english_text: 翻訳する英語テキスト
        
        Returns:
            日本語訳
        """
        # プロンプトを改善（精度と速度のバランス、JSON形式を避ける）
        prompt = f"Translate the following English text to Japanese. Output only the Japanese translation, no explanations, no JSON format, no quotes:\n{english_text}"
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        
        # attention_maskを明示的に設定（警告回避）
        # pad_token_idがNoneの場合は、すべて1のattention_maskを作成
        if self.tokenizer.pad_token_id is None:
            attention_mask = torch.ones_like(inputs).long().to("cuda")
        else:
            attention_mask = (inputs != self.tokenizer.pad_token_id).long().to("cuda")
        
        # 英語テキストの長さに応じて動的に調整（短い文章は高速、長い文章は全文翻訳可能）
        # 日本語は英語の2-3倍のトークン数が必要
        english_words = len(english_text.split())
        # 最小40、最大120トークン（推論速度とのバランス）
        # 注意: 120トークンにすると推論時間が約2-3倍になる可能性があります
        max_new_tokens = min(120, max(40, english_words * 2))
        
        # 推論パラメータの最適化（翻訳は短いのでmax_new_tokensを動的に調整）
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,  # attention_maskを明示的に設定
                max_new_tokens=max_new_tokens,  # 動的に調整（短い文章は高速、長い文章は全文翻訳可能）
                temperature=0.1,  # 0.2→0.1に削減（より確定的で高速）
                top_p=0.8,  # 0.9→0.8に削減（高速化）
                top_k=20,  # 30→20に削減（高速化）
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # KVキャッシュを有効化
                num_beams=1,  # ビームサーチを無効化（高速化）
                repetition_penalty=1.05,
                early_stopping=False  # num_beams=1の時はFalse（警告回避）
            )
        translation = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        # 余分な説明文を削除
        translation = translation.strip()
        
        # JSON形式を削除（{'text': "..."}のような形式）
        import re
        # JSON形式のパターンを検出して削除
        json_pattern = r"\{['\"]text['\"]\s*:\s*['\"](.*?)['\"]\s*\}"
        match = re.search(json_pattern, translation)
        if match:
            translation = match.group(1)
        
        # その他のJSON形式のパターン
        if translation.startswith("{") and translation.endswith("}"):
            try:
                import json
                json_obj = json.loads(translation)
                if isinstance(json_obj, dict) and "text" in json_obj:
                    translation = json_obj["text"]
            except:
                pass
        
        # "翻訳:"や"Translation:"などのプレフィックスを削除
        for prefix in ["翻訳:", "Translation:", "日本語訳:", "Japanese translation:"]:
            if translation.startswith(prefix):
                translation = translation[len(prefix):].strip()
        
        # 引用符を削除
        translation = translation.strip('"').strip("'").strip()
        
        return translation

