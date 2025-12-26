"""
JPEG Forensics Module
======================
Quantization tables, double compression, subsampling detection.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import struct

logger = logging.getLogger(__name__)

# Standard JPEG quantization tables for quality estimation
STANDARD_LUMINANCE_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

class JPEGForensics:
    """JPEG forensic analysis"""
    
    def analyze(self, image_bytes: bytes, image: Image.Image) -> Dict[str, Any]:
        """
        Comprehensive JPEG forensic analysis
        """
        result = {
            "format": self._detect_format(image_bytes, image),
            "subsampling": "unknown",
            "quant_tables_fingerprint": None,
            "estimated_quality": None,
            "double_compression_probability": 0.0,
            "recompress_chain_explanation": [],
            "artifacts": []
        }
        
        # Only analyze JPEG
        if result["format"] != "jpeg":
            result["recompress_chain_explanation"].append(
                f"Format is {result['format']}, JPEG forensics not applicable"
            )
            return result
        
        try:
            # Extract quantization tables
            quant_result = self._extract_quant_tables(image_bytes)
            if quant_result:
                result["quant_tables_fingerprint"] = quant_result["fingerprint"]
                result["estimated_quality"] = quant_result["estimated_quality"]
            
            # Detect subsampling
            result["subsampling"] = self._detect_subsampling(image)
            
            # Double compression detection
            dc_result = self._detect_double_compression(image_bytes, image)
            result["double_compression_probability"] = dc_result["probability"]
            result["artifacts"].extend(dc_result.get("artifacts", []))
            
            # Build recompress chain explanation
            result["recompress_chain_explanation"] = self._build_chain_explanation(result)
            
        except Exception as e:
            logger.warning(f"JPEG forensics error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _detect_format(self, image_bytes: bytes, image: Image.Image) -> str:
        """Detect image format from magic bytes"""
        if len(image_bytes) < 12:
            return "unknown"
        
        # JPEG: FF D8 FF
        if image_bytes[:3] == b'\xff\xd8\xff':
            return "jpeg"
        
        # PNG: 89 50 4E 47
        if image_bytes[:4] == b'\x89PNG':
            return "png"
        
        # WebP: RIFF....WEBP
        if image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "webp"
        
        # HEIC/HEIF: ftyp marker
        if b'ftyp' in image_bytes[:32]:
            if b'heic' in image_bytes[:32] or b'heix' in image_bytes[:32]:
                return "heic"
            if b'mif1' in image_bytes[:32]:
                return "heif"
        
        # GIF: GIF87a or GIF89a
        if image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "gif"
        
        # BMP: BM
        if image_bytes[:2] == b'BM':
            return "bmp"
        
        # Fallback to PIL
        if image.format:
            return image.format.lower()
        
        return "unknown"
    
    def _extract_quant_tables(self, image_bytes: bytes) -> Optional[Dict]:
        """Extract JPEG quantization tables"""
        try:
            tables = []
            pos = 0
            
            while pos < len(image_bytes) - 2:
                # Find DQT marker (FF DB)
                if image_bytes[pos:pos+2] == b'\xff\xdb':
                    pos += 2
                    length = struct.unpack('>H', image_bytes[pos:pos+2])[0]
                    pos += 2
                    
                    # Extract table data
                    table_data = image_bytes[pos:pos+length-2]
                    
                    # Parse table (simplified - assumes 8-bit precision)
                    if len(table_data) >= 65:
                        precision_id = table_data[0]
                        table_values = list(table_data[1:65])
                        tables.append({
                            "id": precision_id & 0x0F,
                            "precision": (precision_id >> 4) & 0x0F,
                            "values": table_values
                        })
                    
                    pos += length - 2
                else:
                    pos += 1
            
            if not tables:
                return None
            
            # Create fingerprint
            fingerprint = self._create_table_fingerprint(tables)
            
            # Estimate quality from luminance table
            quality = self._estimate_quality(tables[0]["values"] if tables else None)
            
            return {
                "fingerprint": fingerprint,
                "estimated_quality": quality,
                "table_count": len(tables)
            }
            
        except Exception as e:
            logger.warning(f"Quant table extraction error: {e}")
            return None
    
    def _create_table_fingerprint(self, tables: List[Dict]) -> str:
        """Create a short fingerprint from quant tables"""
        import hashlib
        
        data = b''
        for table in tables:
            data += bytes(table["values"][:16])  # First 16 values
        
        return hashlib.md5(data).hexdigest()[:12]
    
    def _estimate_quality(self, table_values: List[int]) -> Optional[int]:
        """Estimate JPEG quality from quantization table"""
        if not table_values or len(table_values) < 64:
            return None
        
        try:
            # Reshape to 8x8
            table = np.array(table_values[:64]).reshape(8, 8)
            
            # Compare with standard table at different qualities
            # Quality estimation using average quantization value
            avg_quant = np.mean(table)
            
            # Rough estimation (inverse relationship)
            if avg_quant <= 2:
                return 100
            elif avg_quant >= 100:
                return max(1, int(5000 / avg_quant))
            else:
                # Linear interpolation for mid-range
                quality = int(100 - (avg_quant - 2) * 0.8)
                return max(1, min(100, quality))
                
        except Exception:
            return None
    
    def _detect_subsampling(self, image: Image.Image) -> str:
        """Detect chroma subsampling"""
        try:
            if image.mode != 'RGB':
                return "unknown"
            
            # Convert to YCbCr and check component sizes
            # This is a heuristic based on image characteristics
            width, height = image.size
            
            # Most AI images use 4:4:4 (no subsampling)
            # Most camera JPEGs use 4:2:0
            
            # Check for subsampling artifacts
            img_array = np.array(image)
            
            # Analyze Cb/Cr channels for blocking
            # (simplified heuristic)
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                # Check color channel variance patterns
                r_var = np.var(img_array[:, :, 0])
                g_var = np.var(img_array[:, :, 1])
                b_var = np.var(img_array[:, :, 2])
                
                # 4:2:0 typically shows more color channel smoothing
                color_ratio = (r_var + b_var) / (2 * g_var + 1)
                
                if color_ratio < 0.7:
                    return "4:2:0"
                elif color_ratio > 1.2:
                    return "4:4:4"
                else:
                    return "4:2:2"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def _detect_double_compression(self, image_bytes: bytes, image: Image.Image) -> Dict:
        """Detect double JPEG compression"""
        result = {
            "probability": 0.0,
            "artifacts": [],
            "evidence": []
        }
        
        try:
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            gray = np.mean(img_array, axis=2)
            
            # Method 1: DCT coefficient histogram analysis
            # Double compression creates periodic patterns in DCT histograms
            
            # Analyze 8x8 block boundaries
            h, w = gray.shape
            block_boundary_diffs = []
            
            # Horizontal boundaries
            for i in range(8, h - 8, 8):
                diff = np.abs(gray[i, :] - gray[i-1, :]).mean()
                block_boundary_diffs.append(diff)
            
            # Vertical boundaries
            for j in range(8, w - 8, 8):
                diff = np.abs(gray[:, j] - gray[:, j-1]).mean()
                block_boundary_diffs.append(diff)
            
            if block_boundary_diffs:
                avg_boundary = np.mean(block_boundary_diffs)
                std_boundary = np.std(block_boundary_diffs)
                
                # Strong block artifacts suggest compression
                if avg_boundary > 2.5:
                    result["artifacts"].append("visible_block_artifacts")
                    result["probability"] += 0.2
                
                # Periodic pattern in boundaries suggests double compression
                if std_boundary < avg_boundary * 0.4 and avg_boundary > 1.5:
                    result["artifacts"].append("periodic_block_pattern")
                    result["probability"] += 0.3
                    result["evidence"].append("DCT block periodicity detected")
            
            # Method 2: Histogram analysis of pixel values
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
            
            # Double compression creates "comb" pattern in histogram
            # Check for periodic peaks
            hist_diff = np.diff(hist.astype(float))
            sign_changes = np.sum(np.abs(np.diff(np.sign(hist_diff + 0.001)))) / 2
            
            expected_changes = len(hist_diff) * 0.3  # Random noise level
            if sign_changes > expected_changes * 1.5:
                result["artifacts"].append("histogram_comb_pattern")
                result["probability"] += 0.25
                result["evidence"].append("Histogram shows compression artifacts")
            
            # Cap probability
            result["probability"] = min(0.95, result["probability"])
            
        except Exception as e:
            logger.warning(f"Double compression detection error: {e}")
        
        return result
    
    def _build_chain_explanation(self, result: Dict) -> List[str]:
        """Build human-readable recompression chain explanation"""
        explanations = []
        
        quality = result.get("estimated_quality")
        dc_prob = result.get("double_compression_probability", 0)
        subsampling = result.get("subsampling", "unknown")
        
        if quality:
            if quality >= 95:
                explanations.append(f"High quality JPEG (Q≈{quality}) - minimal compression loss")
            elif quality >= 80:
                explanations.append(f"Good quality JPEG (Q≈{quality}) - standard compression")
            elif quality >= 60:
                explanations.append(f"Medium quality JPEG (Q≈{quality}) - noticeable compression")
            else:
                explanations.append(f"Low quality JPEG (Q≈{quality}) - heavy compression, likely re-saved multiple times")
        
        if dc_prob > 0.5:
            explanations.append(f"High probability of double compression ({dc_prob:.0%}) - image likely re-saved")
        elif dc_prob > 0.25:
            explanations.append(f"Moderate double compression indicators ({dc_prob:.0%})")
        
        if subsampling == "4:2:0":
            explanations.append("4:2:0 subsampling - typical of camera/standard JPEG")
        elif subsampling == "4:4:4":
            explanations.append("4:4:4 subsampling - no chroma subsampling (common in AI/graphics)")
        
        if not explanations:
            explanations.append("No significant compression artifacts detected")
        
        return explanations


def analyze_jpeg_forensics(image_bytes: bytes, image: Image.Image) -> Dict[str, Any]:
    """Convenience function"""
    forensics = JPEGForensics()
    return forensics.analyze(image_bytes, image)
