"""
Hugging Faceキャッシュをクリアするスクリプト
不完全なダウンロードファイルを削除します
"""
import os
import shutil
from pathlib import Path

def clear_model_cache(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """
    特定のモデルのキャッシュをクリア
    
    Args:
        model_name: クリアするモデル名
    """
    # Hugging Faceキャッシュディレクトリ
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_dir.exists():
        print(f"キャッシュディレクトリが見つかりません: {cache_dir}")
        return
    
    print(f"キャッシュディレクトリ: {cache_dir}")
    
    # モデル名からディレクトリ名を生成
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = cache_dir / model_dir_name
    
    if model_cache_path.exists():
        print(f"\nモデルキャッシュを削除中: {model_cache_path}")
        try:
            shutil.rmtree(model_cache_path)
            print("✅ キャッシュを削除しました")
        except Exception as e:
            print(f"❌ エラー: {e}")
    else:
        print(f"\nモデルキャッシュが見つかりません: {model_cache_path}")
    
    # 不完全なダウンロードファイルを探す
    print("\n不完全なダウンロードファイルを検索中...")
    incomplete_files = []
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.safetensors.incomplete') or file.endswith('.tmp'):
                incomplete_files.append(os.path.join(root, file))
    
    if incomplete_files:
        print(f"\n不完全なファイルを {len(incomplete_files)} 個見つけました:")
        for f in incomplete_files[:10]:  # 最初の10個を表示
            print(f"  - {f}")
        
        response = input("\nこれらのファイルを削除しますか？ (y/n): ")
        if response.lower() == 'y':
            for f in incomplete_files:
                try:
                    os.remove(f)
                    print(f"✅ 削除: {os.path.basename(f)}")
                except Exception as e:
                    print(f"❌ エラー ({os.path.basename(f)}): {e}")
    else:
        print("不完全なファイルは見つかりませんでした")

if __name__ == "__main__":
    print("=" * 60)
    print("Hugging Face キャッシュクリアツール")
    print("=" * 60)
    
    model_name = input("\nクリアするモデル名を入力 (Enterでデフォルト: meta-llama/Llama-3.1-8B-Instruct): ").strip()
    if not model_name:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    clear_model_cache(model_name)
    
    print("\n" + "=" * 60)
    print("完了しました。再度 python scripts/app.py を実行してください。")
    print("=" * 60)

