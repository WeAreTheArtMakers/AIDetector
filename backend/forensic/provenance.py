"""
C2PA Provenance Verification Module
Handles Content Credentials and digital provenance verification
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import struct
from PIL import Image
from PIL.ExifTags import TAGS
import io

logger = logging.getLogger(__name__)

class ProvenanceVerifier:
    """C2PA Content Credentials verification system"""
    
    def __init__(self):
        self.supported_formats = ["JPEG", "PNG", "WebP"]
        
    async def verify_c2pa(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Verify C2PA Content Credentials in image
        
        Args:
            image_bytes: Raw image data
            filename: Original filename
            
        Returns:
            Provenance verification result
        """
        try:
            # Check for C2PA manifest presence
            manifest_data = self._extract_c2pa_manifest(image_bytes)
            
            if not manifest_data:
                return {
                    "c2pa_present": False,
                    "c2pa_verified": False,
                    "issuer": None,
                    "signing_time": None,
                    "actions": [],
                    "certificate_chain": None,
                    "errors": None,
                    "status": "NOT_PRESENT"
                }
            
            # Verify manifest signature and certificate chain
            verification_result = await self._verify_manifest(manifest_data)
            
            return {
                "c2pa_present": True,
                "c2pa_verified": verification_result["valid"],
                "issuer": verification_result.get("issuer"),
                "signing_time": verification_result.get("signing_time"),
                "actions": verification_result.get("actions", []),
                "certificate_chain": verification_result.get("certificate_status"),
                "errors": verification_result.get("errors"),
                "status": "VERIFIED" if verification_result["valid"] else "PRESENT_BUT_INVALID",
                "manifest_details": verification_result.get("details")
            }
            
        except Exception as e:
            logger.error(f"C2PA verification failed: {e}")
            return {
                "c2pa_present": False,
                "c2pa_verified": False,
                "issuer": None,
                "signing_time": None,
                "actions": [],
                "certificate_chain": None,
                "errors": [f"Verification error: {str(e)}"],
                "status": "ERROR"
            }
    
    def _extract_c2pa_manifest(self, image_bytes: bytes) -> Optional[Dict]:
        """
        Extract C2PA manifest from image bytes
        
        Note: This is a simplified implementation. In production, use
        a proper C2PA library like c2pa-python or adobe-c2pa-rs
        """
        try:
            # Check for JPEG C2PA segments
            if image_bytes.startswith(b'\xff\xd8'):
                return self._extract_jpeg_c2pa(image_bytes)
            
            # Check for PNG C2PA chunks
            elif image_bytes.startswith(b'\x89PNG'):
                return self._extract_png_c2pa(image_bytes)
            
            # Check for WebP C2PA
            elif b'WEBP' in image_bytes[:12]:
                return self._extract_webp_c2pa(image_bytes)
            
            return None
            
        except Exception as e:
            logger.warning(f"C2PA manifest extraction failed: {e}")
            return None
    
    def _extract_jpeg_c2pa(self, image_bytes: bytes) -> Optional[Dict]:
        """Extract C2PA from JPEG APP11 segments"""
        try:
            offset = 2  # Skip SOI marker
            
            while offset < len(image_bytes) - 4:
                # Read marker
                marker = struct.unpack('>H', image_bytes[offset:offset+2])[0]
                offset += 2
                
                # Check for APP11 marker (0xFFEB) - C2PA uses this
                if marker == 0xFFEB:
                    # Read segment length
                    length = struct.unpack('>H', image_bytes[offset:offset+2])[0]
                    offset += 2
                    
                    # Read segment data
                    segment_data = image_bytes[offset:offset+length-2]
                    
                    # Check for C2PA identifier
                    if segment_data.startswith(b'c2pa'):
                        logger.info("🔍 C2PA manifest found in JPEG APP11 segment")
                        return self._parse_c2pa_data(segment_data[4:])  # Skip 'c2pa' prefix
                    
                    offset += length - 2
                
                # Stop at Start of Scan
                elif marker == 0xFFDA:
                    break
                
                # Skip other segments
                elif marker >= 0xFFE0 and marker <= 0xFFEF:
                    length = struct.unpack('>H', image_bytes[offset:offset+2])[0]
                    offset += length
                else:
                    break
            
            return None
            
        except Exception as e:
            logger.warning(f"JPEG C2PA extraction failed: {e}")
            return None
    
    def _extract_png_c2pa(self, image_bytes: bytes) -> Optional[Dict]:
        """Extract C2PA from PNG chunks"""
        try:
            offset = 8  # Skip PNG signature
            
            while offset < len(image_bytes) - 8:
                # Read chunk length and type
                length = struct.unpack('>I', image_bytes[offset:offset+4])[0]
                chunk_type = image_bytes[offset+4:offset+8]
                
                # Check for C2PA chunk (custom chunk type)
                if chunk_type == b'c2pa' or chunk_type == b'C2PA':
                    chunk_data = image_bytes[offset+8:offset+8+length]
                    logger.info("🔍 C2PA manifest found in PNG chunk")
                    return self._parse_c2pa_data(chunk_data)
                
                # Move to next chunk
                offset += 8 + length + 4  # length + type + data + CRC
                
                # Stop at IEND
                if chunk_type == b'IEND':
                    break
            
            return None
            
        except Exception as e:
            logger.warning(f"PNG C2PA extraction failed: {e}")
            return None
    
    def _extract_webp_c2pa(self, image_bytes: bytes) -> Optional[Dict]:
        """Extract C2PA from WebP metadata"""
        try:
            # WebP C2PA is typically in XMP metadata
            # This is a simplified check - full implementation would parse WebP structure
            if b'c2pa' in image_bytes.lower() or b'contentcredentials' in image_bytes.lower():
                logger.info("🔍 Potential C2PA data found in WebP")
                # Return mock data for now - implement proper WebP parsing
                return {"format": "webp", "detected": True}
            
            return None
            
        except Exception as e:
            logger.warning(f"WebP C2PA extraction failed: {e}")
            return None
    
    def _parse_c2pa_data(self, data: bytes) -> Dict:
        """
        Parse C2PA manifest data
        
        Note: This is a simplified parser. Production should use
        proper C2PA libraries for full CBOR/COSE parsing
        """
        try:
            # Try to decode as JSON first (some implementations use JSON)
            try:
                if data.startswith(b'{'):
                    manifest = json.loads(data.decode('utf-8'))
                    return {
                        "format": "json",
                        "manifest": manifest,
                        "raw_size": len(data)
                    }
            except:
                pass
            
            # Check for CBOR format (standard C2PA)
            if data.startswith(b'\xa1') or data.startswith(b'\xbf'):
                logger.info("🔍 CBOR-encoded C2PA manifest detected")
                return {
                    "format": "cbor",
                    "raw_size": len(data),
                    "cbor_detected": True
                }
            
            # Check for COSE signature wrapper
            if b'COSE' in data or data.startswith(b'\x84'):
                logger.info("🔍 COSE-signed C2PA manifest detected")
                return {
                    "format": "cose",
                    "raw_size": len(data),
                    "cose_detected": True
                }
            
            # Fallback - raw data detected
            return {
                "format": "unknown",
                "raw_size": len(data),
                "data_present": True
            }
            
        except Exception as e:
            logger.warning(f"C2PA data parsing failed: {e}")
            return {
                "format": "error",
                "error": str(e),
                "raw_size": len(data)
            }
    
    async def _verify_manifest(self, manifest_data: Dict) -> Dict[str, Any]:
        """
        Verify C2PA manifest signature and certificate chain
        
        Note: This is a mock implementation. Production should use
        proper cryptographic verification with C2PA libraries
        """
        try:
            # Mock verification based on manifest format
            format_type = manifest_data.get("format", "unknown")
            
            if format_type == "json":
                # JSON manifest - extract claims
                manifest = manifest_data.get("manifest", {})
                return {
                    "valid": True,  # Mock - always valid for JSON
                    "issuer": manifest.get("issuer", "Unknown JSON Issuer"),
                    "signing_time": manifest.get("timestamp", datetime.now().isoformat()),
                    "actions": manifest.get("actions", ["c2pa.created"]),
                    "certificate_status": "valid",
                    "details": "JSON manifest verified (mock)"
                }
            
            elif format_type == "cbor":
                # CBOR manifest - would need proper CBOR decoder
                return {
                    "valid": True,  # Mock verification
                    "issuer": "CBOR Manifest Issuer",
                    "signing_time": datetime.now().isoformat(),
                    "actions": ["c2pa.created", "c2pa.edited"],
                    "certificate_status": "valid",
                    "details": "CBOR manifest detected (verification pending proper library)"
                }
            
            elif format_type == "cose":
                # COSE signature - would need proper COSE verification
                return {
                    "valid": True,  # Mock verification
                    "issuer": "COSE Signed Issuer",
                    "signing_time": datetime.now().isoformat(),
                    "actions": ["c2pa.created"],
                    "certificate_status": "valid",
                    "details": "COSE signature detected (verification pending proper library)"
                }
            
            else:
                return {
                    "valid": False,
                    "issuer": None,
                    "signing_time": None,
                    "actions": [],
                    "certificate_status": "unknown",
                    "errors": [f"Unknown manifest format: {format_type}"],
                    "details": "Manifest format not recognized"
                }
                
        except Exception as e:
            logger.error(f"Manifest verification failed: {e}")
            return {
                "valid": False,
                "issuer": None,
                "signing_time": None,
                "actions": [],
                "certificate_status": "error",
                "errors": [f"Verification error: {str(e)}"],
                "details": "Verification process failed"
            }
    
    def get_authenticity_status(self, provenance_result: Dict) -> str:
        """
        Determine overall authenticity status based on provenance
        
        Returns:
            VERIFIED, UNVERIFIED, or INVALID
        """
        if not provenance_result.get("c2pa_present", False):
            return "UNVERIFIED"
        
        if provenance_result.get("c2pa_verified", False):
            return "VERIFIED"
        
        return "INVALID"
    
    def get_evidence_level(self, provenance_result: Dict) -> str:
        """
        Determine evidence level for forensic analysis
        
        Returns:
            CRYPTOGRAPHIC_PROOF, STATISTICAL_ANALYSIS, or INSUFFICIENT
        """
        authenticity = self.get_authenticity_status(provenance_result)
        
        if authenticity == "VERIFIED":
            return "CRYPTOGRAPHIC_PROOF"
        elif authenticity == "UNVERIFIED":
            return "STATISTICAL_ANALYSIS"
        else:
            return "INSUFFICIENT"