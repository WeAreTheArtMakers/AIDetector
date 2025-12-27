"""
Text Forensics Module - OCR + AI Text Artifact Detection
==========================================================
Detects text in images and analyzes for AI generation artifacts.

AI-generated images often have:
- Nonsensical/gibberish text
- Inconsistent fonts within same sign
- Malformed letters (extra strokes, missing parts)
- Repeated characters or patterns
- Text that doesn't match any language

This module:
1. Detects text regions using OCR (EasyOCR or Tesseract)
2. Analyzes detected text for AI artifacts
3. Provides AI evidence score based on text quality

Dependencies (optional, graceful fallback):
- easyocr: pip install easyocr (recommended, better accuracy)
- pytesseract: pip install pytesseract (fallback, requires tesseract binary)
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy imports for OCR libraries
_easyocr_reader = None
_easyocr_available = None
_tesseract_available = None


def _check_easyocr_available() -> bool:
    """Check if EasyOCR is available"""
    global _easyocr_available
    if _easyocr_available is not None:
        return _easyocr_available
    
    try:
        import easyocr
        _easyocr_available = True
        logger.info("EasyOCR available")
    except ImportError:
        _easyocr_available = False
        logger.warning("EasyOCR not available (pip install easyocr)")
    
    return _easyocr_available


def _check_tesseract_available() -> bool:
    """Check if Tesseract is available"""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    
    try:
        import pytesseract
        # Try to run tesseract to verify it's installed
        pytesseract.get_tesseract_version()
        _tesseract_available = True
        logger.info("Tesseract available")
    except Exception:
        _tesseract_available = False
        logger.warning("Tesseract not available")
    
    return _tesseract_available


def _get_easyocr_reader():
    """Lazy load EasyOCR reader with multiple languages"""
    global _easyocr_reader
    
    if _easyocr_reader is not None:
        return _easyocr_reader
    
    if not _check_easyocr_available():
        return None
    
    try:
        import easyocr
        # Support multiple languages for better detection
        # en: English, tr: Turkish, es: Spanish (for European plates)
        # de: German, fr: French, it: Italian, pt: Portuguese
        _easyocr_reader = easyocr.Reader(
            ['en', 'tr', 'es', 'de', 'fr', 'it', 'pt'], 
            gpu=False, 
            verbose=False
        )
        logger.info("EasyOCR reader initialized (en, tr, es, de, fr, it, pt)")
        return _easyocr_reader
    except Exception as e:
        logger.error(f"Failed to initialize EasyOCR: {e}")
        # Fallback to minimal languages
        try:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized (en only - fallback)")
            return _easyocr_reader
        except Exception as e2:
            logger.error(f"EasyOCR fallback also failed: {e2}")
            return None


@dataclass
class TextRegion:
    """Detected text region"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    # Analysis results
    is_valid_word: bool = False
    is_gibberish: bool = False
    has_malformed_chars: bool = False
    language_detected: str = ""
    ai_artifact_score: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass 
class TextForensicsResult:
    """Result of text forensics analysis"""
    enabled: bool = True
    has_text: bool = False
    text_count: int = 0
    
    # Detected text regions
    regions: List[TextRegion] = field(default_factory=list)
    
    # Overall scores
    ai_text_score: float = 0.0  # 0-1, higher = more AI-like text
    gibberish_ratio: float = 0.0  # Ratio of gibberish text
    valid_word_ratio: float = 0.0  # Ratio of valid words
    
    # Evidence
    ai_text_evidence: List[str] = field(default_factory=list)
    real_text_evidence: List[str] = field(default_factory=list)
    
    # Summary
    verdict: str = "no_text"  # no_text, likely_real, likely_ai, inconclusive
    confidence: str = "low"


# Common English words for validation
COMMON_ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    # Common sign words
    'open', 'closed', 'exit', 'enter', 'stop', 'go', 'sale', 'free', 'hotel',
    'restaurant', 'cafe', 'bar', 'shop', 'store', 'market', 'bank', 'pharmacy',
    'hospital', 'police', 'fire', 'parking', 'no', 'yes', 'welcome', 'thank',
    'street', 'road', 'avenue', 'boulevard', 'plaza', 'square', 'park',
    # Numbers as words
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    # Common abbreviations
    'st', 'rd', 'ave', 'blvd', 'dr', 'mr', 'mrs', 'ms', 'jr', 'sr',
}

# Common Turkish words
COMMON_TURKISH_WORDS = {
    've', 'bir', 'bu', 'da', 'de', 'için', 'ile', 'ne', 'var', 'ben',
    'sen', 'o', 'biz', 'siz', 'onlar', 'gibi', 'daha', 'çok', 'en', 'her',
    'ama', 'ya', 'ki', 'mi', 'mu', 'mı', 'mü', 'değil', 'olan', 'olarak',
    # Common sign words
    'açık', 'kapalı', 'giriş', 'çıkış', 'dur', 'dikkat', 'tehlike', 'yasak',
    'otopark', 'market', 'eczane', 'hastane', 'polis', 'itfaiye', 'banka',
    'cadde', 'sokak', 'bulvar', 'meydan', 'park', 'bahçe',
}

# Gibberish patterns (common in AI-generated text)
GIBBERISH_PATTERNS = [
    r'^[^aeiouAEIOUğüşıöçĞÜŞİÖÇ]{4,}$',  # No vowels (4+ chars)
    r'(.)\1{3,}',  # Same character repeated 4+ times
    r'^[A-Z]{5,}$',  # All caps 5+ chars (often gibberish)
    r'[^a-zA-Z0-9\s\-\'\.]{3,}',  # 3+ special chars in a row
    r'^[bcdfghjklmnpqrstvwxyz]{5,}$',  # Only consonants 5+ chars
]


class TextForensics:
    """
    Analyzes text in images for AI generation artifacts.
    
    AI-generated images often have:
    - Nonsensical text on signs/labels
    - Inconsistent letter shapes
    - Text that looks like words but isn't
    """
    
    def __init__(self):
        self.min_confidence = 0.10  # Very low threshold to catch AI-generated gibberish text
        self.min_text_length = 2  # Minimum text length to analyze
        
    def analyze(self, image: Image.Image) -> TextForensicsResult:
        """
        Analyze image for text and AI artifacts.
        
        Args:
            image: PIL Image
            
        Returns:
            TextForensicsResult with detected text and AI scores
        """
        result = TextForensicsResult()
        
        # Check if any OCR engine is available
        ocr_available = _check_easyocr_available() or _check_tesseract_available()
        if not ocr_available:
            result.enabled = False
            result.verdict = "no_ocr"
            result.confidence = "low"
            return result
        
        # Try to detect text
        raw_detections = self._detect_text(image)
        
        if not raw_detections:
            result.verdict = "no_text"
            result.confidence = "high"
            return result
        
        # Filter and analyze detections
        filtered_count = 0
        for text, conf, bbox in raw_detections:
            logger.info(f"OCR detection: '{text}' conf={conf:.2f} len={len(text.strip())}")
            if conf < self.min_confidence:
                logger.info(f"  -> Filtered: low confidence ({conf:.2f} < {self.min_confidence})")
                filtered_count += 1
                continue
            if len(text.strip()) < self.min_text_length:
                logger.info(f"  -> Filtered: too short ({len(text.strip())} < {self.min_text_length})")
                filtered_count += 1
                continue
            
            region = self._analyze_text_region(text, conf, bbox)
            result.regions.append(region)
            logger.info(f"  -> Passed: gibberish={region.is_gibberish}, valid_word={region.is_valid_word}, ai_score={region.ai_artifact_score:.2f}")
        
        if filtered_count > 0:
            logger.info(f"OCR: {len(raw_detections)} detected, {filtered_count} filtered, {len(result.regions)} passed")
        
        if not result.regions:
            result.verdict = "no_text"
            result.confidence = "high"
            return result
        
        result.has_text = True
        result.text_count = len(result.regions)
        
        # Compute overall scores
        self._compute_overall_scores(result)
        
        # Determine verdict
        self._determine_verdict(result)
        
        return result
    
    def _detect_text(self, image: Image.Image) -> List[Tuple[str, float, Tuple]]:
        """
        Detect text in image using available OCR.
        Applies preprocessing for better detection.
        
        Returns list of (text, confidence, bbox) tuples.
        """
        detections = []
        
        # Preprocess image for better OCR
        processed_image = self._preprocess_for_ocr(image)
        img_array = np.array(processed_image)
        
        # Try EasyOCR first (better accuracy)
        reader = _get_easyocr_reader()
        if reader:
            try:
                # Run OCR with lower threshold for more detections
                results = reader.readtext(
                    img_array,
                    paragraph=False,  # Don't merge into paragraphs
                    min_size=10,  # Detect smaller text
                    text_threshold=0.5,  # Lower threshold for text detection
                    low_text=0.3,  # Lower threshold for text region
                    link_threshold=0.3,  # Lower threshold for linking
                    canvas_size=2560,  # Higher resolution for small text
                    mag_ratio=1.5  # Magnify for small text
                )
                for bbox, text, conf in results:
                    # Convert bbox to (x1, y1, x2, y2)
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    bbox_rect = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                    detections.append((text, conf, bbox_rect))
                logger.info(f"EasyOCR detected {len(detections)} text regions")
                return detections
            except Exception as e:
                logger.error(f"EasyOCR error: {e}")
        
        # Fallback to Tesseract
        if _check_tesseract_available():
            try:
                import pytesseract
                # Get detailed data
                data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
                
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    conf = float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0
                    
                    if text and conf > 0:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bbox = (x, y, x + w, y + h)
                        detections.append((text, conf, bbox))
                
                logger.info(f"Tesseract detected {len(detections)} text regions")
                return detections
            except Exception as e:
                logger.error(f"Tesseract error: {e}")
        
        logger.warning("No OCR engine available")
        result = TextForensicsResult()
        result.enabled = False
        return detections
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR detection.
        - Enhance contrast
        - Sharpen edges
        - Upscale if too small
        """
        from PIL import ImageEnhance, ImageFilter
        
        img = image.convert('RGB')
        
        # Upscale small images
        min_dim = min(img.size)
        if min_dim < 800:
            scale = 800 / min_dim
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)  # 30% more contrast
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)  # 50% sharper
        
        # Slight brightness adjustment
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)  # 5% brighter
        
        return img
    
    def _analyze_text_region(self, text: str, confidence: float, 
                             bbox: Tuple) -> TextRegion:
        """Analyze a single text region for AI artifacts."""
        region = TextRegion(
            text=text,
            confidence=confidence,
            bbox=bbox
        )
        
        # Clean text for analysis
        clean_text = text.strip().lower()
        
        # Check if valid word
        region.is_valid_word = self._is_valid_word(clean_text)
        
        # Check for gibberish patterns
        region.is_gibberish = self._is_gibberish(text)
        
        # Check for malformed characters
        region.has_malformed_chars = self._has_malformed_chars(text)
        
        # Detect language
        region.language_detected = self._detect_language(clean_text)
        
        # Compute AI artifact score for this region
        ai_score = 0.0
        
        if region.is_gibberish:
            ai_score += 0.5
            region.issues.append("Gibberish/nonsensical text")
        
        if not region.is_valid_word and len(clean_text) >= 3:
            ai_score += 0.3
            region.issues.append("Not a recognized word")
        
        if region.has_malformed_chars:
            ai_score += 0.2
            region.issues.append("Malformed characters detected")
        
        # Low OCR confidence on clear text can indicate AI artifacts
        if confidence < 0.5 and len(text) >= 4:
            ai_score += 0.15
            region.issues.append("Low OCR confidence")
        
        # Mixed case in unexpected ways
        if self._has_unusual_casing(text):
            ai_score += 0.1
            region.issues.append("Unusual character casing")
        
        region.ai_artifact_score = min(1.0, ai_score)
        
        return region
    
    def _is_valid_word(self, text: str) -> bool:
        """Check if text is a valid word in known languages."""
        clean = text.strip().lower()
        
        # Remove common punctuation
        clean = re.sub(r'[^\w\s]', '', clean)
        
        # Check against word lists
        if clean in COMMON_ENGLISH_WORDS:
            return True
        if clean in COMMON_TURKISH_WORDS:
            return True
        
        # Check if it's a number
        if clean.isdigit():
            return True
        
        # Check if it looks like a proper noun (capitalized)
        if text[0].isupper() and len(text) >= 2:
            # Could be a name or place
            return True
        
        return False
    
    def _is_gibberish(self, text: str) -> bool:
        """Check if text matches gibberish patterns."""
        for pattern in GIBBERISH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check consonant-to-vowel ratio
        vowels = set('aeiouAEIOUğüşıöçĞÜŞİÖÇ')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        v_count = sum(1 for c in text if c in vowels)
        c_count = sum(1 for c in text if c in consonants)
        
        if len(text) >= 4:
            # Normal text has roughly 40% vowels
            if v_count == 0 and c_count >= 4:
                return True
            if c_count > 0 and v_count / (v_count + c_count) < 0.15:
                return True
        
        return False
    
    def _has_malformed_chars(self, text: str) -> bool:
        """Check for malformed or unusual characters."""
        # Check for Unicode replacement characters
        if '\ufffd' in text:
            return True
        
        # Check for unusual Unicode ranges that might indicate OCR errors on AI text
        for char in text:
            code = ord(char)
            # Private use area or unusual symbols
            if 0xE000 <= code <= 0xF8FF:
                return True
            # Combining characters in unexpected places
            if 0x0300 <= code <= 0x036F and text.index(char) == 0:
                return True
        
        return False
    
    def _has_unusual_casing(self, text: str) -> bool:
        """Check for unusual character casing patterns."""
        if len(text) < 3:
            return False
        
        # Check for random caps in middle of word
        # e.g., "hElLo" or "KOEMO" patterns
        if text.isalpha():
            upper_positions = [i for i, c in enumerate(text) if c.isupper()]
            lower_positions = [i for i, c in enumerate(text) if c.islower()]
            
            # Alternating case is suspicious
            if len(upper_positions) >= 2 and len(lower_positions) >= 2:
                # Check if they alternate
                all_positions = sorted(upper_positions + lower_positions)
                alternating = True
                for i in range(len(all_positions) - 1):
                    if (all_positions[i] in upper_positions) == (all_positions[i+1] in upper_positions):
                        alternating = False
                        break
                if alternating and len(text) >= 4:
                    return True
        
        return False
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        clean = text.lower()
        
        # Turkish-specific characters
        turkish_chars = set('ğüşıöçĞÜŞİÖÇ')
        if any(c in turkish_chars for c in text):
            return "tr"
        
        # Check word lists
        if clean in COMMON_TURKISH_WORDS:
            return "tr"
        if clean in COMMON_ENGLISH_WORDS:
            return "en"
        
        return "unknown"
    
    def _compute_overall_scores(self, result: TextForensicsResult):
        """Compute overall scores from individual regions."""
        if not result.regions:
            return
        
        total_ai_score = 0.0
        gibberish_count = 0
        valid_word_count = 0
        
        for region in result.regions:
            total_ai_score += region.ai_artifact_score
            if region.is_gibberish:
                gibberish_count += 1
            if region.is_valid_word:
                valid_word_count += 1
        
        n = len(result.regions)
        result.ai_text_score = total_ai_score / n
        result.gibberish_ratio = gibberish_count / n
        result.valid_word_ratio = valid_word_count / n
        
        # Build evidence lists
        for region in result.regions:
            if region.ai_artifact_score >= 0.5:
                result.ai_text_evidence.append(
                    f"'{region.text}': {', '.join(region.issues)}"
                )
            elif region.is_valid_word:
                result.real_text_evidence.append(
                    f"'{region.text}': Valid word ({region.language_detected})"
                )
    
    def _determine_verdict(self, result: TextForensicsResult):
        """Determine overall verdict based on text analysis."""
        if not result.has_text:
            result.verdict = "no_text"
            result.confidence = "high"
            return
        
        # Strong AI indicators
        if result.gibberish_ratio >= 0.5:
            result.verdict = "likely_ai"
            result.confidence = "high"
            return
        
        if result.ai_text_score >= 0.6:
            result.verdict = "likely_ai"
            result.confidence = "medium"
            return
        
        # Strong real indicators
        if result.valid_word_ratio >= 0.8 and result.ai_text_score < 0.2:
            result.verdict = "likely_real"
            result.confidence = "medium"
            return
        
        if result.valid_word_ratio >= 0.5 and result.gibberish_ratio == 0:
            result.verdict = "likely_real"
            result.confidence = "low"
            return
        
        # Mixed signals
        result.verdict = "inconclusive"
        result.confidence = "low"


# Global instance
_text_forensics: Optional[TextForensics] = None

def get_text_forensics() -> TextForensics:
    global _text_forensics
    if _text_forensics is None:
        _text_forensics = TextForensics()
    return _text_forensics
