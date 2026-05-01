# AI English Conversation Practice (音声英会話)

マイクまたはテキストで英語のやり取りをし、音声フィードバックと日本語訳・語彙ヒントを得られるデスクトップ向けの学習アプリです。Gradio のローカル UI で動作します。

## 機能

- **音声認識**: マイク入力を英語テキストへ（Faster-Whisper）
- **会話応答**: ローカル LLM による英語の返答
- **音声合成**: Windows 環境では PowerShell 経由の TTS（追加モデル不要）
- **日本語訳・単語**: LLM と spaCy を利用
- **テキスト入力**: マイクなしでも利用可能
- **オフライン運用**: モデルをキャッシュに置けば、`LLM_LOCAL_ONLY` を適切に設定してオフライン利用できます

## 動作環境（目安）

- **OS**: Windows 10/11（TTS は Windows 向け実装）
- **Python**: 3.10 以上（3.13 でも利用可）
- **GPU**: NVIDIA GPU と CUDA 対応の PyTorch を推奨。`LLM_PRESET` ごとの VRAM の目安（概算・量子化の有無や同時負荷で変動します）。デスクトップ向けの一例を併記します。
  - `**light`（約1.5B）**: 比較的控えめ。VRAM **約6GB級**でも運用しやすいことが多い。例: **GeForce RTX 3060（12GB）**
  - `**balanced`（約3B）**: 中程度。**約8GB級以上**があると余裕が出やすい。例: **GeForce RTX 4070 SUPER（12GB）**
  - `**heavy`（約8B）**: 負荷が高い。**約12〜16GB級以上**を推奨し、量子化なしだとさらに欲しくなる場合がある。例: **GeForce RTX 4090（24GB）**
- **メモリ・ストレージ**: 利用する LLM と Whisper のサイズに応じて十分な RAM とディスク空きを確保してください（`heavy` はキャッシュで **十数GB規模** になることが多い）

CUDA の有無は次で確認できます。

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## セットアップ

### 1. 仮想環境（推奨）

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. 依存関係

```bash
pip install -r requirements.txt
```

`bitsandbytes` は環境によって未導入の場合があります。未使用時は主に float16 で読み込み、VRAM 使用量が増えることがあります。

### 3. spaCy（英語）

```bash
python -m spacy download en_core_web_sm
```

### 4. Hugging Face ログイン（Llama を使う場合）

**Qwen**（本プロジェクトの `light` プリセット）は、通常 **モデルページでの個別の利用申請は不要**で取得できます（一般公開のモデルカードに従ってください）。

**Meta Llama**（`balanced` / `heavy`）は **Hugging Face 上で各モデルのライセンスに同意（利用申請）し、承認されたアカウント**で取得する必要があります。手順の概要:

1. 対象モデル（例: [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)、[Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)）のページで申請・承認を得る
2. 端末で `hf auth login`（または `huggingface-cli login`）し、**承認済みアカウント**でログインする

申請なしで試す場合は `LLM_PRESET=light`（Qwen）を指定してください。

### 5. 起動

初回に Hub からモデルを取得する場合は `**LLM_LOCAL_ONLY=false`** を付けてください（既定はキャッシュのみ参照のため）。

```powershell
$env:LLM_LOCAL_ONLY="false"
python app.py
```

キャッシュが済んだあと、オフライン運用では `LLM_LOCAL_ONLY=true`（または未設定のまま）にできます。

ブラウザで `http://127.0.0.1:7860` を開きます。

## LLM の選び方（プリセット）

**環境変数**で指定してください。


| 変数               | 説明                                                          |
| ---------------- | ----------------------------------------------------------- |
| `LLM_PRESET`     | `light` / `balanced`（既定） / `heavy`                          |
| `LLM_LOCAL_ONLY` | `true`（既定）: **キャッシュのみ**読み込み。`false`: キャッシュを試し、無ければ Hub から取得 |



| プリセット          | モデル（例）                             | 利用申請（ゲート）         |
| -------------- | ---------------------------------- | ----------------- |
| `light`        | `Qwen/Qwen2.5-1.5B-Instruct`       | 通常は不要             |
| `balanced`（既定） | `meta-llama/Llama-3.2-3B-Instruct` | **Meta / HF で必要** |
| `heavy`        | `meta-llama/Llama-3.1-8B-Instruct` | **Meta / HF で必要** |


例（軽量・申請不要）:

```powershell
$env:LLM_PRESET="light"
$env:LLM_LOCAL_ONLY="false"
python app.py
```

例（高品質・8B、要承認）:

```powershell
$env:LLM_PRESET="heavy"
$env:LLM_LOCAL_ONLY="false"
python app.py
```

### モデルファイルの保存場所（調査用）

Hugging Face Hub のキャッシュは通常、次のディレクトリにあります（Windows の例）。

```
%USERPROFILE%\.cache\huggingface\hub
```

リポジトリごとに `models--組織名--モデル名` 形式のフォルダができます。

## 使い方

1. マイク録音またはテキストで英語を入力
2. 必要に応じて「日本語訳」「難しい単語」を有効化
3. 「会話履歴をクリア」で文脈をリセット

## プロジェクト構成（抜粋）

```
e_conversation/
├── app.py
├── speech_recognition.py
├── llm_handler.py
├── text_to_speech.py
├── translation.py
├── requirements.txt
└── README.md
```

## トラブルシューティング

- **403 / gated（Llama）**: モデルページで利用申請が承認されているか、`hf auth login` 済みか確認するか、`LLM_PRESET=light` で Qwen に切り替えてください。
- **キャッシュに無いと言われる（初回）**: `LLM_LOCAL_ONLY=false` で起動して取得してください。
- **CUDA / メモリ**: より小さいプリセット、または Whisper の `model_size` を小さくする（`speech_recognition.py`）と負荷が下がります。
- **音声**: 出力デバイスや `sounddevice` のデバイス一覧を確認してください。

## ライセンス

MIT License。詳細は [LICENSE](LICENSE) を参照してください。

## 技術スタック（概要）

- Faster-Whisper、Transformers、Gradio、spaCy など（バージョンは `requirements.txt` を参照）

## 参考リンク

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio](https://www.gradio.app/docs)

