"""
Forensic Metadata Extraction Module
====================================
Comprehensive EXIF, XMP, IPTC, GPS, and metadata analysis.
Supports HEIC with graceful degradation.
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import io
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# Known camera manufacturers
KNOWN_CAMERA_MAKERS = {
    "canon", "nikon", "sony", "fujifilm", "panasonic", "olympus", "leica",
    "pentax", "samsung", "lg", "apple", "google", "huawei", "xiaomi", "oppo",
    "vivo", "oneplus", "motorola", "nokia", "asus", "gopro", "dji", "hasselblad",
    "ricoh", "sigma", "zeiss", "kodak", "casio", "minolta"
}

# AI software signatures
AI_SOFTWARE_SIGNATURES = {
    "dall-e": {"vendor": "OpenAI", "type": "diffusion"},
    "dall·e": {"vendor": "OpenAI", "type": "diffusion"},
    "midjourney": {"vendor": "Midjourney", "type": "diffusion"},
    "stable diffusion": {"vendor": "Stability AI", "type": "diffusion"},
    "sdxl": {"vendor": "Stability AI", "type": "diffusion"},
    "firefly": {"vendor": "Adobe", "type": "diffusion"},
    "imagen": {"vendor": "Google", "type": "diffusion"},
    "leonardo": {"vendor": "Leonardo.AI", "type": "diffusion"},
    "flux": {"vendor": "Black Forest Labs", "type": "diffusion"},
    "ideogram": {"vendor": "Ideogram", "type": "diffusion"},
    "runway": {"vendor": "Runway", "type": "video_diffusion"},
    "nightcafe": {"vendor": "NightCafe", "type": "diffusion"},
    "artbreeder": {"vendor": "Artbreeder", "type": "gan"},
    "deepai": {"vendor": "DeepAI", "type": "diffusion"},
}

class MetadataExtractor:
    """Comprehensive metadata extraction with XMP/IPTC support"""
    
    def extract_all(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract all metadata from image"""
        result = {
            "exif": self._extract_exif(image_bytes),
            "xmp": self._extract_xmp(image_bytes),
            "iptc": self._extract_iptc(image_bytes),
            "gps": self._extract_gps(image_bytes),
            "camera": {},
            "software": {},
            "software_history": [],
            "ai_detection": {"detected": False, "software": None},
            "timestamps": {},
            "technical": {},
            "format_info": {}
        }
        
        # Detect format
        result["format_info"] = self._detect_format(image_bytes)
        
        # Process camera info
        exif = result["exif"]
        result["camera"] = {
            "make": exif.get("Make"),
            "model": exif.get("Model"),
            "lens": exif.get("LensModel"),
            "is_known_maker": self._is_known_camera(exif.get("Make", "")),
            "has_camera_info": bool(exif.get("Make") or exif.get("Model"))
        }
        
        # Process software
        software = exif.get("Software", "")
        result["software"] = {
            "name": software,
            "is_editing_software": self._is_editing_software(software),
        }
        
        # Build software history from XMP
        result["software_history"] = self._build_software_history(result)
        
        # Check for AI software
        ai_check = self._detect_ai_software(exif, result.get("xmp", {}))
        result["ai_detection"] = ai_check
        
        # Timestamps
        result["timestamps"] = {
            "original": exif.get("DateTimeOriginal"),
            "digitized": exif.get("DateTimeDigitized"),
            "modified": exif.get("DateTime"),
            "gps_timestamp": result["gps"].get("timestamp_utc"),
            "xmp_create": result.get("xmp", {}).get("CreateDate"),
            "xmp_modify": result.get("xmp", {}).get("ModifyDate")
        }
        
        # Technical
        result["technical"] = {
            "exposure_time": exif.get("ExposureTime"),
            "f_number": exif.get("FNumber"),
            "iso": exif.get("ISOSpeedRatings"),
            "focal_length": exif.get("FocalLength"),
            "flash": exif.get("Flash"),
            "white_balance": exif.get("WhiteBalance"),
            "color_space": exif.get("ColorSpace"),
            "orientation": exif.get("Orientation")
        }
        
        # Calculate completeness score
        result["completeness_score"] = self._calculate_completeness(result)
        
        return result
    
    def _detect_format(self, image_bytes: bytes) -> Dict:
        """Detect image format with HEIC support"""
        result = {
            "format": "unknown",
            "heic_supported": False,
            "notes": []
        }
        
        if len(image_bytes) < 12:
            return result
        
        # JPEG
        if image_bytes[:3] == b'\xff\xd8\xff':
            result["format"] = "jpeg"
        # PNG
        elif image_bytes[:4] == b'\x89PNG':
            result["format"] = "png"
        # WebP
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            result["format"] = "webp"
        # HEIC/HEIF
        elif b'ftyp' in image_bytes[:32]:
            if b'heic' in image_bytes[:32] or b'heix' in image_bytes[:32]:
                result["format"] = "heic"
                result["notes"].append("HEIC format detected (iPhone)")
                # Check if pillow-heif is available
                try:
                    import pillow_heif
                    result["heic_supported"] = True
                except ImportError:
                    result["heic_supported"] = False
                    result["notes"].append("HEIC reading not supported - install pillow-heif")
            elif b'mif1' in image_bytes[:32]:
                result["format"] = "heif"
        # GIF
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            result["format"] = "gif"
        
        return result

    
    def _extract_exif(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract EXIF data"""
        exif_data = {}
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            if hasattr(image, '_getexif') and image._getexif():
                raw_exif = image._getexif()
                
                for tag_id, value in raw_exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    
                    # Skip GPS (handled separately) and binary data
                    if tag_name == "GPSInfo":
                        continue
                    if tag_name == "MakerNote":
                        continue
                    
                    # Convert to serializable format
                    try:
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='ignore')
                        elif hasattr(value, 'numerator'):  # Rational
                            value = float(value)
                        else:
                            value = str(value)
                        exif_data[tag_name] = value
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"EXIF extraction error: {e}")
            
        return exif_data
    
    def _extract_xmp(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract XMP metadata"""
        xmp_data = {}
        
        try:
            # Find XMP packet in image bytes
            xmp_start = image_bytes.find(b'<x:xmpmeta')
            if xmp_start == -1:
                xmp_start = image_bytes.find(b'<?xpacket')
            
            if xmp_start != -1:
                xmp_end = image_bytes.find(b'</x:xmpmeta>', xmp_start)
                if xmp_end == -1:
                    xmp_end = image_bytes.find(b'<?xpacket end', xmp_start)
                
                if xmp_end != -1:
                    xmp_bytes = image_bytes[xmp_start:xmp_end + 20]
                    xmp_str = xmp_bytes.decode('utf-8', errors='ignore')
                    
                    # Parse key fields
                    xmp_data = self._parse_xmp_string(xmp_str)
                    
        except Exception as e:
            logger.warning(f"XMP extraction error: {e}")
        
        return xmp_data
    
    def _parse_xmp_string(self, xmp_str: str) -> Dict[str, Any]:
        """Parse XMP string for key fields"""
        result = {}
        
        # Common XMP fields to extract
        patterns = {
            "CreatorTool": r'xmp:CreatorTool["\s>]+([^<"]+)',
            "CreateDate": r'xmp:CreateDate["\s>]+([^<"]+)',
            "ModifyDate": r'xmp:ModifyDate["\s>]+([^<"]+)',
            "MetadataDate": r'xmp:MetadataDate["\s>]+([^<"]+)',
            "DerivedFrom": r'stRef:documentID["\s>]+([^<"]+)',
            "HistoryAction": r'stEvt:action["\s>]+([^<"]+)',
            "HistorySoftware": r'stEvt:softwareAgent["\s>]+([^<"]+)',
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, xmp_str, re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()
        
        # Check for AI-related tools
        ai_patterns = ["midjourney", "dall-e", "stable diffusion", "firefly", "leonardo"]
        xmp_lower = xmp_str.lower()
        for ai_tool in ai_patterns:
            if ai_tool in xmp_lower:
                result["ai_tool_detected"] = ai_tool
                break
        
        return result
    
    def _extract_iptc(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract IPTC metadata with proper encoding and binary detection"""
        iptc_data = {}
        
        try:
            # IPTC marker: 0x1C (photoshop 3.0)
            # Look for IPTC-IIM markers
            pos = 0
            while pos < len(image_bytes) - 5:
                if image_bytes[pos:pos+2] == b'\x1c\x02':
                    # Found IPTC record
                    record_type = image_bytes[pos + 2]
                    size = (image_bytes[pos + 3] << 8) + image_bytes[pos + 4]
                    
                    if size > 0 and pos + 5 + size <= len(image_bytes):
                        value_bytes = image_bytes[pos + 5:pos + 5 + size]
                        
                        # Map common IPTC tags
                        iptc_tags = {
                            5: "ObjectName",
                            25: "Keywords",
                            55: "DateCreated",
                            80: "Byline",
                            85: "BylineTitle",
                            90: "City",
                            101: "Country",
                            105: "Headline",
                            116: "Copyright",
                            120: "Caption"
                        }
                        
                        if record_type in iptc_tags:
                            tag_name = iptc_tags[record_type]
                            parsed = self._parse_iptc_value(value_bytes, tag_name)
                            if parsed["valid"]:
                                iptc_data[tag_name] = parsed["value"]
                            elif parsed.get("status"):
                                iptc_data[f"{tag_name}_status"] = parsed["status"]
                    
                    pos += 5 + size
                else:
                    pos += 1
                    
        except Exception as e:
            logger.warning(f"IPTC extraction error: {e}")
        
        return iptc_data
    
    def _parse_iptc_value(self, value_bytes: bytes, tag_name: str) -> Dict[str, Any]:
        """
        Parse IPTC value with encoding detection and binary validation.
        
        Returns:
            Dict with 'valid', 'value', 'status', 'encoding'
        """
        result = {
            "valid": False,
            "value": None,
            "status": None,
            "encoding": None
        }
        
        if not value_bytes or len(value_bytes) == 0:
            result["status"] = "empty"
            return result
        
        # Check printable ratio
        printable_count = sum(1 for b in value_bytes if 32 <= b <= 126 or b in (9, 10, 13))
        printable_ratio = printable_count / len(value_bytes)
        
        # If less than 85% printable, treat as binary/invalid
        if printable_ratio < 0.85:
            result["status"] = "invalid_or_binary"
            return result
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                decoded = value_bytes.decode(encoding)
                
                # Validate decoded string
                if self._is_valid_text(decoded):
                    # Apply max length limit (500 chars)
                    if len(decoded) > 500:
                        decoded = decoded[:500] + "..."
                    
                    result["valid"] = True
                    result["value"] = decoded.strip()
                    result["encoding"] = encoding
                    return result
                    
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        result["status"] = "decode_failed"
        return result
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if decoded text is valid (not garbage)"""
        if not text or len(text.strip()) == 0:
            return False
        
        # Check for excessive control characters
        control_count = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
        if control_count > len(text) * 0.1:
            return False
        
        # Check for excessive replacement characters
        if text.count('\ufffd') > len(text) * 0.1:
            return False
        
        return True
    
    def _build_software_history(self, metadata: Dict) -> List[str]:
        """Build software history from all metadata sources"""
        history = []
        
        # From EXIF
        exif_software = metadata.get("exif", {}).get("Software")
        if exif_software:
            history.append(f"EXIF Software: {exif_software}")
        
        # From XMP
        xmp = metadata.get("xmp", {})
        if xmp.get("CreatorTool"):
            history.append(f"XMP CreatorTool: {xmp['CreatorTool']}")
        if xmp.get("HistorySoftware"):
            history.append(f"XMP History: {xmp['HistorySoftware']}")
        if xmp.get("DerivedFrom"):
            history.append(f"XMP DerivedFrom: {xmp['DerivedFrom']}")
        
        # From IPTC
        iptc = metadata.get("iptc", {})
        if iptc.get("Byline"):
            history.append(f"IPTC Byline: {iptc['Byline']}")
        
        return history
    
    def _extract_gps(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract GPS data with full validation
        Returns structured GPS info with confidence assessment
        """
        gps_result = {
            "present": False,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "timestamp_utc": None,
            "source_tags": [],
            "confidence": "none",
            "notes": [],
            "raw": {}
        }
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            if not hasattr(image, '_getexif') or not image._getexif():
                gps_result["notes"].append("No EXIF data found")
                return gps_result
            
            exif = image._getexif()
            
            # GPS tag ID is 34853
            gps_info = exif.get(34853)
            if not gps_info:
                gps_result["notes"].append("No GPS tags in EXIF")
                return gps_result
            
            gps_result["present"] = True
            
            # Parse GPS tags
            gps_data = {}
            for tag_id, value in gps_info.items():
                tag_name = GPSTAGS.get(tag_id, str(tag_id))
                gps_data[tag_name] = value
                gps_result["source_tags"].append(tag_name)
            
            gps_result["raw"] = {k: str(v) for k, v in gps_data.items()}
            
            # Extract latitude
            lat = self._parse_gps_coordinate(
                gps_data.get("GPSLatitude"),
                gps_data.get("GPSLatitudeRef", "N")
            )
            
            # Extract longitude
            lon = self._parse_gps_coordinate(
                gps_data.get("GPSLongitude"),
                gps_data.get("GPSLongitudeRef", "E")
            )
            
            # Extract altitude
            alt = self._parse_altitude(
                gps_data.get("GPSAltitude"),
                gps_data.get("GPSAltitudeRef", 0)
            )
            
            # Extract timestamp
            timestamp = self._parse_gps_timestamp(
                gps_data.get("GPSDateStamp"),
                gps_data.get("GPSTimeStamp")
            )
            
            gps_result["latitude"] = lat
            gps_result["longitude"] = lon
            gps_result["altitude_m"] = alt
            gps_result["timestamp_utc"] = timestamp
            
            # Validate and assess confidence
            gps_result = self._validate_gps(gps_result)
            
        except Exception as e:
            logger.warning(f"GPS extraction error: {e}")
            gps_result["notes"].append(f"Extraction error: {str(e)}")
            
        return gps_result
    
    def _parse_gps_coordinate(self, dms_tuple, ref: str) -> Optional[float]:
        """Convert DMS (degrees, minutes, seconds) to decimal degrees"""
        if not dms_tuple:
            return None
            
        try:
            # Handle different formats
            if isinstance(dms_tuple, (list, tuple)) and len(dms_tuple) >= 3:
                degrees = float(dms_tuple[0])
                minutes = float(dms_tuple[1])
                seconds = float(dms_tuple[2])
            else:
                return None
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            # Apply reference (N/S for lat, E/W for lon)
            if ref in ('S', 'W'):
                decimal = -decimal
                
            return round(decimal, 6)
            
        except Exception as e:
            logger.warning(f"GPS coordinate parse error: {e}")
            return None
    
    def _parse_altitude(self, alt_value, alt_ref) -> Optional[float]:
        """Parse GPS altitude"""
        if alt_value is None:
            return None
            
        try:
            if hasattr(alt_value, 'numerator'):
                alt = float(alt_value)
            else:
                alt = float(alt_value)
            
            # Below sea level if ref is 1
            if alt_ref == 1:
                alt = -alt
                
            return round(alt, 1)
            
        except:
            return None
    
    def _parse_gps_timestamp(self, date_stamp, time_stamp) -> Optional[str]:
        """Parse GPS timestamp to ISO format"""
        if not date_stamp or not time_stamp:
            return None
            
        try:
            # Date format: "YYYY:MM:DD"
            date_parts = str(date_stamp).split(":")
            
            # Time format: (hour, minute, second) as rationals
            if isinstance(time_stamp, (list, tuple)) and len(time_stamp) >= 3:
                hour = int(float(time_stamp[0]))
                minute = int(float(time_stamp[1]))
                second = int(float(time_stamp[2]))
            else:
                return None
            
            dt = datetime(
                int(date_parts[0]), int(date_parts[1]), int(date_parts[2]),
                hour, minute, second
            )
            
            return dt.isoformat() + "Z"
            
        except Exception as e:
            logger.warning(f"GPS timestamp parse error: {e}")
            return None
    
    def _validate_gps(self, gps: Dict) -> Dict:
        """Validate GPS data and assess confidence"""
        lat = gps.get("latitude")
        lon = gps.get("longitude")
        
        if lat is None or lon is None:
            gps["confidence"] = "none"
            gps["notes"].append("Incomplete GPS coordinates")
            return gps
        
        # Validate ranges
        if not (-90 <= lat <= 90):
            gps["confidence"] = "invalid"
            gps["notes"].append(f"Invalid latitude: {lat}")
            return gps
            
        if not (-180 <= lon <= 180):
            gps["confidence"] = "invalid"
            gps["notes"].append(f"Invalid longitude: {lon}")
            return gps
        
        # Check for suspicious values
        suspicious = False
        
        # Null Island (0, 0)
        if abs(lat) < 0.01 and abs(lon) < 0.01:
            suspicious = True
            gps["notes"].append("Coordinates near 0,0 (Null Island) - suspicious")
        
        # Exact round numbers
        if lat == int(lat) and lon == int(lon):
            suspicious = True
            gps["notes"].append("Exact round coordinates - possibly fake")
        
        # Assess confidence
        if suspicious:
            gps["confidence"] = "low"
        elif gps.get("timestamp_utc"):
            gps["confidence"] = "high"
        else:
            gps["confidence"] = "medium"
            gps["notes"].append("No GPS timestamp")
        
        # Always add warning about EXIF editability
        gps["notes"].append("EXIF GPS can be edited/removed - treat as clue, not proof")
        
        return gps

    
    def _is_known_camera(self, make: str) -> bool:
        """Check if camera maker is known"""
        if not make:
            return False
        return any(m in make.lower() for m in KNOWN_CAMERA_MAKERS)
    
    def _is_editing_software(self, software: str) -> bool:
        """Check if software is legitimate editing software"""
        if not software:
            return False
        editing_software = [
            "photoshop", "lightroom", "capture one", "gimp", "affinity",
            "darktable", "rawtherapee", "luminar", "photos", "preview",
            "snapseed", "vsco", "instagram"
        ]
        return any(s in software.lower() for s in editing_software)
    
    def _detect_ai_software(self, exif: Dict, xmp: Dict = None) -> Dict[str, Any]:
        """Detect AI generation software in metadata"""
        result = {
            "detected": False,
            "software": None,
            "vendor": None,
            "type": None,
            "found_in": None
        }
        
        xmp = xmp or {}
        
        # Check XMP first (more detailed)
        if xmp.get("ai_tool_detected"):
            tool = xmp["ai_tool_detected"]
            if tool in AI_SOFTWARE_SIGNATURES:
                info = AI_SOFTWARE_SIGNATURES[tool]
                result["detected"] = True
                result["software"] = tool.title()
                result["vendor"] = info["vendor"]
                result["type"] = info["type"]
                result["found_in"] = "XMP metadata"
                return result
        
        # Fields to search
        search_fields = ["Software", "Creator", "Artist", "Copyright", 
                        "ImageDescription", "UserComment", "XPComment"]
        
        all_text = ""
        for field in search_fields:
            value = exif.get(field, "")
            if value:
                all_text += f" {str(value).lower()}"
        
        # Add XMP fields
        for key, value in xmp.items():
            if isinstance(value, str):
                all_text += f" {value.lower()}"
        
        # Search for AI signatures
        for signature, info in AI_SOFTWARE_SIGNATURES.items():
            if signature in all_text:
                result["detected"] = True
                result["software"] = signature.title()
                result["vendor"] = info["vendor"]
                result["type"] = info["type"]
                result["found_in"] = "EXIF/XMP metadata"
                logger.info(f"🚨 AI software detected: {signature}")
                break
        
        return result
    
    def _calculate_completeness(self, metadata: Dict) -> int:
        """Calculate metadata completeness score (0-100)"""
        score = 0
        max_score = 100
        
        # EXIF presence (20 points)
        if metadata["exif"]:
            score += 10
            if len(metadata["exif"]) > 10:
                score += 10
        
        # Camera info (25 points)
        if metadata["camera"]["has_camera_info"]:
            score += 15
            if metadata["camera"]["is_known_maker"]:
                score += 10
        
        # GPS (20 points)
        if metadata["gps"]["present"]:
            score += 10
            if metadata["gps"]["confidence"] in ["high", "medium"]:
                score += 10
        
        # Timestamps (15 points)
        if metadata["timestamps"]["original"]:
            score += 15
        
        # Technical settings (20 points)
        tech = metadata["technical"]
        tech_fields = ["exposure_time", "f_number", "iso", "focal_length"]
        tech_count = sum(1 for f in tech_fields if tech.get(f))
        score += tech_count * 5
        
        return min(score, max_score)


def extract_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """Convenience function for metadata extraction"""
    extractor = MetadataExtractor()
    return extractor.extract_all(image_bytes)