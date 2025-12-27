"""
Provenance Scoring Module - EXIF/GPS Evidence Weighting
=========================================================
Computes provenance_score for forensic verdict generation.

Key principles:
- EXIF/GPS is "supporting clue", NOT definitive proof
- Provenance boosts photo posterior, never directly sets "proof"
- Missing camera make/model should NOT collapse camera_score
- Platform software (Instagram) is re-encoding, NOT AI generation

Scoring components:
P1) EXIF presence completeness
P2) Consistency checks (plausibility, not blind trust)
P3) Domain interaction (exif_stripped vs recompressed)
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


# Platform software that re-encodes but doesn't indicate AI
PLATFORM_SOFTWARE = {
    "instagram", "whatsapp", "facebook", "twitter", "snapchat",
    "tiktok", "telegram", "signal", "messenger", "wechat",
    "line", "viber", "kakaotalk", "discord"
}

# Editing software (legitimate photo editing)
EDITING_SOFTWARE = {
    "photoshop", "lightroom", "capture one", "gimp", "affinity",
    "darktable", "rawtherapee", "luminar", "photos", "preview",
    "snapseed", "vsco", "pixelmator", "acorn", "pixlr"
}


class ProvenanceScorer:
    """
    Computes provenance_score from EXIF/GPS/metadata.
    
    Output:
    - provenance_score ∈ [0, 1]
    - provenance_rationale: List[str] (TR/EN)
    - provenance_flags: Dict with detailed breakdown
    """
    
    def __init__(self):
        # Scoring weights
        self.weights = {
            "exif_fields_10plus": 0.15,
            "datetime_original": 0.10,
            "subsec_time": 0.10,
            "gps_present": 0.15,
            "gps_timestamp": 0.05,
            "offset_time": 0.05,
            "camera_make_model": 0.05,
            "lens_info": 0.03,
            "exposure_settings": 0.05,
            "platform_software_bonus": 0.07,  # Instagram etc. = real photo re-encoded
            "old_timestamp_bonus": 0.12  # NEW: Pre-2020 timestamp bonus
        }
        
        # Penalties
        self.penalties = {
            "metadata_inconsistency": -0.20,
            "suspicious_gps": -0.15,
            "impossible_values": -0.25,
            "synthetic_patterns": -0.20
        }
    
    def compute(self, 
                metadata: Dict[str, Any],
                domain: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compute provenance score from metadata.
        
        Args:
            metadata: Full metadata dict from MetadataExtractor
            domain: Domain detection result
        
        Returns:
            Dict with:
            - score: float [0, 1]
            - rationale_tr: List[str]
            - rationale_en: List[str]
            - flags: Dict with detailed breakdown
            - is_platform_reencoded: bool
        """
        result = {
            "score": 0.0,
            "rationale_tr": [],
            "rationale_en": [],
            "flags": {
                "exif_present": False,
                "exif_count": 0,
                "datetime_original": False,
                "gps_present": False,
                "gps_valid": False,
                "camera_info": False,
                "platform_software": None,
                "is_platform_reencoded": False,
                "consistency_issues": [],
                "penalties_applied": []
            },
            "is_platform_reencoded": False
        }
        
        if not metadata:
            result["rationale_tr"].append("Metadata bulunamadı")
            result["rationale_en"].append("No metadata found")
            return result
        
        domain = domain or {}
        characteristics = domain.get("characteristics", [])
        
        # Check if EXIF is stripped
        if "exif_stripped" in characteristics:
            result["flags"]["exif_stripped"] = True
            result["rationale_tr"].append("EXIF verisi silinmiş — provenance skoru 0")
            result["rationale_en"].append("EXIF data stripped — provenance score 0")
            return result
        
        score = 0.0
        
        # === P1: EXIF Presence Completeness ===
        exif = metadata.get("exif", {})
        exif_count = len(exif)
        result["flags"]["exif_count"] = exif_count
        
        if exif_count > 0:
            result["flags"]["exif_present"] = True
        
        # +0.15 if >= 10 EXIF fields
        if exif_count >= 10:
            score += self.weights["exif_fields_10plus"]
            result["rationale_tr"].append(f"EXIF alanları mevcut ({exif_count} alan)")
            result["rationale_en"].append(f"EXIF fields present ({exif_count} fields)")
        
        # +0.10 if DateTimeOriginal exists
        timestamps = metadata.get("timestamps", {})
        if timestamps.get("original"):
            score += self.weights["datetime_original"]
            result["flags"]["datetime_original"] = True
            result["rationale_tr"].append("DateTimeOriginal mevcut")
            result["rationale_en"].append("DateTimeOriginal present")
            
            # === NEW: Check for old timestamp (pre-2020) ===
            old_timestamp = self._check_old_timestamp(timestamps)
            if old_timestamp:
                score += self.weights["old_timestamp_bonus"]
                result["flags"]["old_timestamp"] = True
                result["rationale_tr"].append("Eski tarih (2020 öncesi) — AI yaygınlaşmadan önce")
                result["rationale_en"].append("Old timestamp (pre-2020) — before AI became widespread")
        
        # +0.10 if SubSecTime exists
        if exif.get("SubSecTimeOriginal") or exif.get("SubSecTime"):
            score += self.weights["subsec_time"]
            result["rationale_tr"].append("SubSecTime mevcut (milisaniye hassasiyeti)")
            result["rationale_en"].append("SubSecTime present (millisecond precision)")
        
        # +0.15 if GPS lat/lon exists
        gps = metadata.get("gps", {})
        if gps.get("present") and gps.get("latitude") is not None:
            score += self.weights["gps_present"]
            result["flags"]["gps_present"] = True
            result["rationale_tr"].append("GPS koordinatları mevcut")
            result["rationale_en"].append("GPS coordinates present")
            
            # +0.05 if GPS timestamp exists
            if gps.get("timestamp_utc"):
                score += self.weights["gps_timestamp"]
                result["rationale_tr"].append("GPS zaman damgası mevcut")
                result["rationale_en"].append("GPS timestamp present")
            
            # Validate GPS
            gps_valid, gps_issues = self._validate_gps(gps)
            result["flags"]["gps_valid"] = gps_valid
            if not gps_valid:
                score += self.penalties["suspicious_gps"]
                result["flags"]["penalties_applied"].append("suspicious_gps")
                result["flags"]["consistency_issues"].extend(gps_issues)
                for issue in gps_issues:
                    result["rationale_tr"].append(f"GPS şüpheli: {issue}")
                    result["rationale_en"].append(f"GPS suspicious: {issue}")
        
        # +0.05 if OffsetTime fields exist (timezone info)
        if exif.get("OffsetTime") or exif.get("OffsetTimeOriginal"):
            score += self.weights["offset_time"]
            result["rationale_tr"].append("Saat dilimi bilgisi mevcut")
            result["rationale_en"].append("Timezone info present")
        
        # +0.05 if camera make/model exists (but don't penalize absence heavily)
        camera = metadata.get("camera", {})
        if camera.get("has_camera_info"):
            score += self.weights["camera_make_model"]
            result["flags"]["camera_info"] = True
            make = camera.get("make", "")
            model = camera.get("model", "")
            result["rationale_tr"].append(f"Kamera bilgisi: {make} {model}".strip())
            result["rationale_en"].append(f"Camera info: {make} {model}".strip())
        
        # +0.03 if lens info exists
        if camera.get("lens"):
            score += self.weights["lens_info"]
            result["rationale_tr"].append(f"Lens bilgisi: {camera.get('lens')}")
            result["rationale_en"].append(f"Lens info: {camera.get('lens')}")
        
        # +0.05 if exposure settings exist
        technical = metadata.get("technical", {})
        exposure_fields = ["exposure_time", "f_number", "iso"]
        exposure_count = sum(1 for f in exposure_fields if technical.get(f))
        if exposure_count >= 2:
            score += self.weights["exposure_settings"]
            result["rationale_tr"].append("Pozlama ayarları mevcut")
            result["rationale_en"].append("Exposure settings present")
        
        # === P2: Platform Software Detection ===
        software = metadata.get("software", {})
        software_name = software.get("name", "").lower() if software else ""
        
        platform_detected = None
        for platform in PLATFORM_SOFTWARE:
            if platform in software_name:
                platform_detected = platform
                break
        
        if platform_detected:
            result["flags"]["platform_software"] = platform_detected
            result["flags"]["is_platform_reencoded"] = True
            result["is_platform_reencoded"] = True
            # Platform re-encoding is POSITIVE evidence for real photo
            score += self.weights["platform_software_bonus"]
            result["rationale_tr"].append(f"Platform yazılımı tespit edildi: {platform_detected.title()} (yeniden kodlama)")
            result["rationale_en"].append(f"Platform software detected: {platform_detected.title()} (re-encoding)")
        
        # === P3: Consistency Checks ===
        consistency_issues = self._check_consistency(metadata, timestamps, gps)
        if consistency_issues:
            result["flags"]["consistency_issues"].extend(consistency_issues)
            score += self.penalties["metadata_inconsistency"]
            result["flags"]["penalties_applied"].append("metadata_inconsistency")
            for issue in consistency_issues:
                result["rationale_tr"].append(f"Tutarsızlık: {issue}")
                result["rationale_en"].append(f"Inconsistency: {issue}")
        
        # === P4: Synthetic Pattern Detection ===
        synthetic_issues = self._detect_synthetic_patterns(exif, gps)
        if synthetic_issues:
            result["flags"]["consistency_issues"].extend(synthetic_issues)
            score += self.penalties["synthetic_patterns"]
            result["flags"]["penalties_applied"].append("synthetic_patterns")
            for issue in synthetic_issues:
                result["rationale_tr"].append(f"Sentetik desen: {issue}")
                result["rationale_en"].append(f"Synthetic pattern: {issue}")
        
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        result["score"] = round(score, 3)
        
        # Add summary rationale
        if score >= 0.45:
            result["rationale_tr"].insert(0, "EXIF/GPS: Fotoğraf lehine destekleyici ipucu (doğrulanabilirlik sınırlı: metadata değiştirilebilir)")
            result["rationale_en"].insert(0, "EXIF/GPS: Supporting clue for camera photo (limited verifiability: metadata can be altered)")
        elif score >= 0.20:
            result["rationale_tr"].insert(0, "EXIF/GPS: Kısmi provenance verisi mevcut")
            result["rationale_en"].insert(0, "EXIF/GPS: Partial provenance data present")
        else:
            result["rationale_tr"].insert(0, "EXIF/GPS: Yetersiz provenance verisi")
            result["rationale_en"].insert(0, "EXIF/GPS: Insufficient provenance data")
        
        logger.info(f"Provenance score: {score:.3f}, platform={platform_detected}, "
                   f"gps={gps.get('present')}, datetime={timestamps.get('original') is not None}")
        
        return result
    
    def _validate_gps(self, gps: Dict) -> Tuple[bool, List[str]]:
        """
        Validate GPS data for suspicious patterns.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        lat = gps.get("latitude")
        lon = gps.get("longitude")
        
        if lat is None or lon is None:
            return False, ["Missing coordinates"]
        
        # Check for Null Island (0, 0)
        if abs(lat) < 0.01 and abs(lon) < 0.01:
            issues.append("Null Island (0,0) coordinates")
        
        # Check for exact round numbers (suspicious)
        if lat == int(lat) and lon == int(lon):
            issues.append("Exact round coordinates")
        
        # Check for impossible values
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            issues.append("Out of range coordinates")
        
        # Check for suspiciously precise values (too many decimals can indicate fake)
        lat_str = str(lat)
        lon_str = str(lon)
        if '.' in lat_str and len(lat_str.split('.')[1]) > 8:
            issues.append("Suspiciously high precision")
        
        return len(issues) == 0, issues
    
    def _check_consistency(self, metadata: Dict, timestamps: Dict, gps: Dict) -> List[str]:
        """
        Check metadata consistency.
        
        Returns list of inconsistency issues.
        """
        issues = []
        
        # Check timestamp consistency
        dt_original = timestamps.get("original")
        dt_modified = timestamps.get("modified")
        gps_timestamp = gps.get("timestamp_utc")
        
        if dt_original and dt_modified:
            try:
                # Parse dates (format: "YYYY:MM:DD HH:MM:SS")
                orig = self._parse_exif_datetime(dt_original)
                mod = self._parse_exif_datetime(dt_modified)
                
                if orig and mod:
                    # Modified should be >= original
                    if mod < orig:
                        issues.append("Modified date before original date")
                    
                    # Check for epoch dates (1970-01-01)
                    if orig.year < 1990:
                        issues.append("Original date before 1990 (suspicious)")
                    
                    # Check for future dates
                    if orig.year > datetime.now().year + 1:
                        issues.append("Original date in future")
            except:
                pass
        
        # Check GPS timestamp vs photo timestamp
        if gps_timestamp and dt_original:
            try:
                gps_dt = datetime.fromisoformat(gps_timestamp.replace('Z', '+00:00'))
                orig_dt = self._parse_exif_datetime(dt_original)
                
                if gps_dt and orig_dt:
                    # GPS and photo time should be within 24 hours
                    diff = abs((gps_dt.replace(tzinfo=None) - orig_dt).total_seconds())
                    if diff > 86400:  # 24 hours
                        issues.append("GPS timestamp differs from photo timestamp by >24h")
            except:
                pass
        
        return issues
    
    def _detect_synthetic_patterns(self, exif: Dict, gps: Dict) -> List[str]:
        """
        Detect patterns that suggest synthetic/fake metadata.
        
        Returns list of synthetic pattern issues.
        """
        issues = []
        
        # Check for repeated zeros in GPS
        if gps.get("present"):
            lat = gps.get("latitude", 0)
            lon = gps.get("longitude", 0)
            
            # Check for suspicious patterns like 12.000000
            lat_str = f"{lat:.6f}"
            lon_str = f"{lon:.6f}"
            
            if lat_str.endswith("000000") or lon_str.endswith("000000"):
                issues.append("GPS coordinates with suspicious zero pattern")
        
        # Check for default/template values in EXIF
        software = exif.get("Software", "")
        if software:
            # Some fake EXIF tools leave signatures
            fake_signatures = ["exiftool", "fake", "test", "sample"]
            if any(sig in software.lower() for sig in fake_signatures):
                issues.append("Suspicious software signature in EXIF")
        
        # Check for missing expected fields when others are present
        # (Real cameras usually have consistent field sets)
        has_exposure = exif.get("ExposureTime") is not None
        has_fnumber = exif.get("FNumber") is not None
        has_iso = exif.get("ISOSpeedRatings") is not None
        
        # If has some exposure fields but missing others, could be synthetic
        exposure_fields = [has_exposure, has_fnumber, has_iso]
        if sum(exposure_fields) == 1:
            issues.append("Incomplete exposure data (partial fields)")
        
        return issues
    
    def _parse_exif_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse EXIF datetime string to datetime object."""
        if not dt_str:
            return None
        
        try:
            # Standard EXIF format: "YYYY:MM:DD HH:MM:SS"
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except:
            try:
                # Alternative format
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except:
                return None
    
    def _check_old_timestamp(self, timestamps: Dict) -> bool:
        """
        Check if timestamp is from before AI became widespread (pre-2020).
        
        Returns True if timestamp is from 2019 or earlier.
        """
        dt_original = timestamps.get("original", "")
        if dt_original:
            try:
                # Parse EXIF format: "YYYY:MM:DD HH:MM:SS"
                year = int(dt_original[:4])
                if year <= 2019:
                    return True
            except:
                pass
        
        return False


# Global instance
_provenance_scorer: Optional[ProvenanceScorer] = None

def get_provenance_scorer() -> ProvenanceScorer:
    global _provenance_scorer
    if _provenance_scorer is None:
        _provenance_scorer = ProvenanceScorer()
    return _provenance_scorer
