import re
import string
import pandas as pd
import numpy as np
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from typing import List, Dict, Any, Optional

# Import logger from dependencies.py
try:
    from .dependencies import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Could not import 'logger' from dependencies.py. Using fallback logging configuration.")


class IndonesianTextPreprocessor:
    """Indonesian text preprocessing class with comprehensive cleaning"""

    def __init__(self, use_minimal_preprocessing: bool = True,
                 remove_stopwords: bool = False, use_stemming: bool = False,
                 normalize_slang: bool = True):
        # Initialize Indonesian NLP tools with error handling
        try:
            self.stemmer = StemmerFactory().create_stemmer()
            self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
            logger.info("Sastrawi Stemmer and StopWordRemover initialized.")
        except Exception as e:
            logger.warning(f"⚠️ Warning: Failed to initialize Sastrawi tools: {e}")
            self.stemmer = None
            self.stopword_remover = None

        # Download NLTK data if needed
        self._download_nltk_data()

        # Store preprocessing flags
        self.use_minimal_preprocessing = use_minimal_preprocessing
        self.remove_stopwords_flag = remove_stopwords
        self.use_stemming_flag = use_stemming
        self.normalize_slang_flag = normalize_slang

        # Indonesian stopwords (extended)
        self.indonesian_stopwords = set([
            'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
            'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'tidak', 'ini', 'itu',
            'adalah', 'ada', 'akan', 'atau', 'oleh', 'dari', 'dengan', 'dalam', 'sebagai',
            'kami', 'kita', 'mereka', 'anda', 'saya', 'kamu', 'nya', 'ku', 'mu',
            'di', 'ke', 'dari', 'untuk', 'pada', 'dalam', 'oleh', 'tentang', 'atas', 'bawah',
            'dislike', 'disukai', 'tidak suka', 'suka', 'sangat', 'bisa', 'boleh',
            'shopee', 'tokopedia', 'bukalapak', 'lazada', 'blibli', 'jd.id', 'olx', 'aja',
            'kesini', 'tiap', 'dislikenya', 'tapi', 'saja', 'mereka', 'kita', 'kami',
            'ya', 'apa', 'lu', 'chrome', 'lah', 'kan', 'dong', 'sih', 'kok', 'deh'
        ])
        logger.info(f"Loaded {len(self.indonesian_stopwords)} Indonesian stopwords.")


        # Enhanced slang dictionary for Indonesian
        self.slang_dict = {
            'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'tdk': 'tidak', 'g': 'tidak',
            'yg': 'yang', 'dgn': 'dengan', 'krn': 'karena', 'utk': 'untuk',
            'sy': 'saya', 'km': 'kamu', 'lo': 'kamu', 'gw': 'saya', 'gue': 'saya',
            'bgt': 'banget', 'skrg': 'sekarang', 'org': 'orang', 'jgn': 'jangan',
            'udh': 'sudah', 'blm': 'belum', 'tp': 'tapi', 'klo': 'kalau',
            'kyk': 'seperti', 'gmn': 'bagaimana', 'knp': 'kenapa', 'hrs': 'harus',
            'bs': 'bisa', 'emg': 'memang', 'mksd': 'maksud', 'tau': 'tahu',
            'lg': 'lagi', 'jd': 'jadi', 'sm': 'sama', 'dr': 'dari',
            'ke': 'ke', 'wkt': 'waktu', 'cpt': 'cepat', 'lmyn': 'lumayan'
        }
        logger.info(f"Loaded {len(self.slang_dict)} slang words for normalization.")


    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("NLTK 'punkt' tokenizer found.")
        except LookupError:
            try:
                logger.info("NLTK 'punkt' tokenizer not found, attempting download...")
                nltk.download('punkt', quiet=True)
                logger.info("NLTK 'punkt' tokenizer downloaded successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Warning: Failed to download NLTK punkt: {e}")

    def clean_text(self, text: str) -> str:
        """Basic text cleaning with error handling"""
        if pd.isna(text) or text is None:
            return ""

        try:
            text = str(text).lower().strip()

            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

            # Remove mentions and hashtags
            text = re.sub(r'@\\w+', '', text)
            text = re.sub(r'#\\w+', '', text)

            # Remove email addresses
            text = re.sub(r'\\S+@\\S+', '', text)

            # Remove phone numbers
            text = re.sub(r'\\b\\d{10,15}\\b', '', text)

            # Remove excessive punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{2,}', '.', text)

            # Remove extra whitespace and newlines
            text = re.sub(r'\\s+', ' ', text).strip()

            return text

        except Exception as e:
            logger.error(f"⚠️ Error in clean_text: {e}", exc_info=True)
            return str(text) if text else ""

    def normalize_slang(self, text: str) -> str:
        """Normalize Indonesian slang words"""
        try:
            words = text.split()
            normalized_words = []

            for word in words:
                # Check if word is in slang dictionary
                if word.lower() in self.slang_dict:
                    normalized_words.append(self.slang_dict[word.lower()])
                else:
                    normalized_words.append(word)

            return ' '.join(normalized_words)

        except Exception as e:
            logger.error(f"⚠️ Error in normalize_slang: {e}", exc_info=True)
            return text

    def remove_punctuation(self, text: str, keep_important: bool = True) -> str:
        """Remove punctuation but optionally keep emotional indicators"""
        try:
            if keep_important:
                # Keep some punctuation that might be important for sentiment
                important_punct = '!?'
                translator = str.maketrans('', '', string.punctuation.replace(important_punct, ''))
                return text.translate(translator)
            else:
                # Remove all punctuation
                translator = str.maketrans('', '', string.punctuation)
                return text.translate(translator)

        except Exception as e:
            logger.error(f"⚠️ Error in remove_punctuation: {e}", exc_info=True)
            return text

    def remove_stopwords(self, text: str) -> str:
        """Remove Indonesian stopwords"""
        try:
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in self.indonesian_stopwords]
            return ' '.join(filtered_words)

        except Exception as e:
            logger.error(f"⚠️ Error in remove_stopwords: {e}", exc_info=True)
            return text

    def stem_text(self, text: str) -> str:
        """Apply stemming using Sastrawi"""
        try:
            if self.stemmer:
                return self.stemmer.stem(text)
            else:
                logger.warning("⚠️ Stemmer not available, returning original text")
                return text

        except Exception as e:
            logger.error(f"⚠️ Error in stem_text: {e}", exc_info=True)
            return text

    def preprocess(self, text: str, use_stemming: Optional[bool] = None,
                   remove_stopwords: Optional[bool] = None, normalize_slang: Optional[bool] = None,
                   remove_punct: bool = True) -> str:
        """
        Complete preprocessing pipeline with parameter control.
        Optional parameters will use initialization values if not provided.
        """
        try:
            # Input validation
            if not text or pd.isna(text):
                return ""

            # Use flags from initialization if not overridden by method call
            _normalize_slang = normalize_slang if normalize_slang is not None else self.normalize_slang_flag
            _remove_stopwords = remove_stopwords if remove_stopwords is not None else self.remove_stopwords_flag
            _use_stemming = use_stemming if use_stemming is not None else self.use_stemming_flag

            # Step 1: Basic cleaning
            text = self.clean_text(text)
            if not text:
                return ""

            # Step 2: Normalize slang (optional)
            if _normalize_slang:
                text = self.normalize_slang(text)

            # Step 3: Remove punctuation (optional)
            if remove_punct: # remove_punct is always True from predictor.py
                text = self.remove_punctuation(text, keep_important=True)

            # Step 4: Remove stopwords (optional - be careful with transformers)
            if _remove_stopwords:
                text = self.remove_stopwords(text)

            # Step 5: Stemming (optional - be very careful with transformers)
            if _use_stemming:
                text = self.stem_text(text)

            # Final cleaning
            text = ' '.join(text.split())  # Remove extra spaces

            return text if text else ""

        except Exception as e:
            logger.error(f"⚠️ Error in preprocess: {e}", exc_info=True)
            return str(text) if text else ""

    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """Batch preprocessing for efficiency"""
        try:
            return [self.preprocess(text, **kwargs) for text in texts]
        except Exception as e:
            logger.error(f"⚠️ Error in batch_preprocess: {e}", exc_info=True)
            return texts

    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """Get text statistics"""
        try:
            words = text.split()
            return {
                'char_count': len(text),
                'word_count': len(words),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'unique_words': len(set(words)),
                'stopword_ratio': sum(1 for word in words if word.lower() in self.indonesian_stopwords) / max(1, len(words))
            }
        except Exception as e:
            logger.error(f"⚠️ Error in get_text_stats: {e}", exc_info=True)
            return {}