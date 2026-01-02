import spacy
from collections import Counter

class TranslationHelper:
    def __init__(self, llm_handler):
        """
        翻訳・単語分析ヘルパーの初期化
        
        Args:
            llm_handler: LLMHandlerインスタンス
        """
        self.llm = llm_handler
        # 英語の難易度分析用
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Please install it with:")
            print("python -m spacy download en_core_web_sm")
            print("Falling back to basic word extraction...")
            self.nlp = None
    
    def translate_to_japanese(self, english_text):
        """
        LLMを使用して英語を日本語に翻訳
        
        Args:
            english_text: 翻訳する英語テキスト
        
        Returns:
            日本語訳
        """
        return self.llm.translate_to_japanese(english_text)
    
    def extract_difficult_words(self, text, difficulty_threshold=5):
        """
        難しい単語を抽出
        
        Args:
            text: 分析するテキスト
            difficulty_threshold: 難易度の閾値（単語の長さなど）
        
        Returns:
            難しい単語のリスト（単語、品詞、見出し語を含む辞書のリスト）
        """
        if self.nlp is None:
            # spaCyが使えない場合の簡易版
            words = text.split()
            difficult_words = []
            seen = set()
            
            for word in words:
                # 記号を除去
                clean_word = ''.join(c for c in word if c.isalpha())
                if len(clean_word) > difficulty_threshold and clean_word.lower() not in seen:
                    seen.add(clean_word.lower())
                    difficult_words.append({
                        "word": clean_word,
                        "pos": "UNKNOWN",
                        "lemma": clean_word.lower()
                    })
            
            return difficult_words
        
        doc = self.nlp(text)
        difficult_words = []
        seen = set()
        
        for token in doc:
            if token.is_alpha and len(token.text) > difficulty_threshold:
                lemma = token.lemma_.lower()
                if lemma not in seen:
                    seen.add(lemma)
                    difficult_words.append({
                        "word": token.text,
                        "pos": token.pos_,
                        "lemma": token.lemma_
                    })
        
        return difficult_words
    
    def get_word_translations(self, words_list):
        """
        単語リストの日本語訳を取得
        
        Args:
            words_list: 単語のリスト（辞書形式）
        
        Returns:
            単語とその日本語訳の辞書
        """
        translations = {}
        for word_info in words_list:
            word = word_info["word"]
            if word not in translations:
                translation = self.llm.translate_to_japanese(word)
                translations[word] = translation
        
        return translations

