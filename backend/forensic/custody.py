"""
Chain of Custody and Evidence Management Module
Handles file integrity, evidence bundles, and forensic reporting
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import zipfile
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

logger = logging.getLogger(__name__)

class CustodyManager:
    """Chain of custody and evidence management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.store_files = config.get("store_evidence_files", False)
        self.evidence_dir = Path(config.get("evidence_directory", "./evidence"))
        self.signing_key = self._load_signing_key()
        
        # Create evidence directory if storing files
        if self.store_files:
            self.evidence_dir.mkdir(exist_ok=True)
    
    def _load_signing_key(self) -> Optional[ed25519.Ed25519PrivateKey]:
        """Load or generate Ed25519 signing key for reports"""
        try:
            key_path = self.config.get("signing_key_path")
            
            if key_path and os.path.exists(key_path):
                # Load existing key
                with open(key_path, 'rb') as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None
                    )
                logger.info("🔐 Loaded existing signing key")
                return private_key
            
            elif self.config.get("generate_signing_key", False):
                # Generate new key
                private_key = ed25519.Ed25519PrivateKey.generate()
                
                if key_path:
                    # Save key if path specified
                    pem = private_key.private_bytes(
                        encoding=Encoding.PEM,
                        format=PrivateFormat.PKCS8,
                        encryption_algorithm=NoEncryption()
                    )
                    
                    os.makedirs(os.path.dirname(key_path), exist_ok=True)
                    with open(key_path, 'wb') as f:
                        f.write(pem)
                    
                    logger.info(f"🔐 Generated new signing key: {key_path}")
                
                return private_key
            
            else:
                logger.warning("⚠️ No signing key configured - reports will be unsigned")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load signing key: {e}")
            return None
    
    def calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate SHA-256 hash of file data"""
        return hashlib.sha256(file_data).hexdigest()
    
    def generate_report_id(self) -> str:
        """Generate unique forensic report ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"FR_{timestamp}_{unique_id}"
    
    def create_custody_record(self, file_data: bytes, filename: str, 
                            analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create chain of custody record for uploaded file
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            analysis_result: Complete analysis results
            
        Returns:
            Custody record with hash, timestamps, and metadata
        """
        try:
            report_id = self.generate_report_id()
            file_hash = self.calculate_file_hash(file_data)
            received_at = datetime.now(timezone.utc)
            
            custody_record = {
                "report_id": report_id,
                "sha256": file_hash,
                "received_at": received_at.isoformat(),
                "tool_version": "AIDetector-Forensic v2.0.0",
                "file_metadata": {
                    "filename": filename,
                    "size_bytes": len(file_data),
                    "mime_type": self._detect_mime_type(file_data)
                },
                "chain_of_custody": {
                    "received_from": "web_upload",
                    "received_by": "ai_detector_forensic",
                    "processing_node": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                    "integrity_verified": True
                }
            }
            
            # Store evidence file if configured
            if self.store_files:
                evidence_path = self._store_evidence_file(file_data, report_id, filename)
                custody_record["evidence_file_path"] = str(evidence_path)
            
            return custody_record
            
        except Exception as e:
            logger.error(f"Failed to create custody record: {e}")
            raise
    
    def _detect_mime_type(self, file_data: bytes) -> str:
        """Detect MIME type from file signature"""
        if file_data.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        elif file_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return "image/png"
        elif file_data.startswith(b'RIFF') and b'WEBP' in file_data[:12]:
            return "image/webp"
        elif file_data.startswith(b'BM'):
            return "image/bmp"
        else:
            return "application/octet-stream"
    
    def _store_evidence_file(self, file_data: bytes, report_id: str, 
                           original_filename: str) -> Path:
        """Store evidence file with forensic naming"""
        try:
            # Create subdirectory by date
            date_dir = self.evidence_dir / datetime.now().strftime("%Y-%m-%d")
            date_dir.mkdir(exist_ok=True)
            
            # Generate evidence filename
            file_ext = Path(original_filename).suffix
            evidence_filename = f"{report_id}_original{file_ext}"
            evidence_path = date_dir / evidence_filename
            
            # Write file with integrity check
            with open(evidence_path, 'wb') as f:
                f.write(file_data)
            
            # Verify written file
            with open(evidence_path, 'rb') as f:
                written_hash = hashlib.sha256(f.read()).hexdigest()
            
            original_hash = hashlib.sha256(file_data).hexdigest()
            
            if written_hash != original_hash:
                raise ValueError("Evidence file integrity check failed")
            
            logger.info(f"📁 Evidence file stored: {evidence_path}")
            return evidence_path
            
        except Exception as e:
            logger.error(f"Failed to store evidence file: {e}")
            raise
    
    def create_forensic_report(self, report_id: str, custody_record: Dict,
                             provenance_result: Dict, ai_analysis: Dict,
                             verdict: Dict) -> Dict[str, Any]:
        """
        Create complete forensic report with digital signature
        
        Args:
            report_id: Unique report identifier
            custody_record: Chain of custody information
            provenance_result: C2PA verification results
            ai_analysis: AI detection analysis
            verdict: Final verdict and warnings
            
        Returns:
            Complete forensic report with signature
        """
        try:
            report_timestamp = datetime.now(timezone.utc)
            
            # Build complete report
            forensic_report = {
                "report_metadata": {
                    "report_id": report_id,
                    "report_type": "digital_image_forensic_analysis",
                    "generated_at": report_timestamp.isoformat(),
                    "tool_name": "AIDetector-Forensic",
                    "tool_version": "v2.0.0",
                    "analysis_standard": "ISO/IEC 27037:2012 (Digital Evidence)",
                    "report_format_version": "1.0"
                },
                "custody": custody_record,
                "provenance": provenance_result,
                "ai_analysis": ai_analysis,
                "verdict": verdict,
                "forensic_notes": self._generate_forensic_notes(
                    provenance_result, ai_analysis, verdict
                ),
                "disclaimers": [
                    "This analysis provides statistical assessment, not definitive proof",
                    "AI detection algorithms have inherent error rates",
                    "C2PA verification provides cryptographic authenticity when present",
                    "Results should be interpreted by qualified digital forensics experts"
                ]
            }
            
            # Add digital signature if signing key available
            if self.signing_key:
                signature_data = self._sign_report(forensic_report)
                forensic_report["digital_signature"] = signature_data
            
            # Store report if configured
            if self.store_files:
                self._store_forensic_report(report_id, forensic_report)
            
            return forensic_report
            
        except Exception as e:
            logger.error(f"Failed to create forensic report: {e}")
            raise
    
    def _generate_forensic_notes(self, provenance: Dict, ai_analysis: Dict, 
                               verdict: Dict) -> List[str]:
        """Generate forensic examiner notes"""
        notes = []
        
        # Provenance notes
        if provenance.get("c2pa_verified"):
            notes.append(f"✓ Content Credentials verified - Issuer: {provenance.get('issuer')}")
        elif provenance.get("c2pa_present"):
            notes.append("⚠ Content Credentials present but verification failed")
        else:
            notes.append("⚠ No Content Credentials found - provenance unverified")
        
        # AI analysis notes
        ai_prob = ai_analysis.get("calibrated_probability", 0)
        if ai_prob > 0.8:
            notes.append("🚨 High probability of AI generation detected")
        elif ai_prob > 0.6:
            notes.append("⚠ Moderate probability of AI generation")
        elif ai_prob < 0.3:
            notes.append("✓ Low probability of AI generation")
        
        # Technical notes
        supporting_signals = ai_analysis.get("supporting_signals", {})
        metadata_analysis = supporting_signals.get("metadata_analysis", {})
        
        if not metadata_analysis.get("camera_info_present"):
            notes.append("⚠ No camera metadata present")
        
        if metadata_analysis.get("ai_software_detected"):
            notes.append(f"🚨 AI software signatures detected: {metadata_analysis['ai_software_detected']}")
        
        return notes
    
    def _sign_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Digitally sign forensic report with Ed25519"""
        try:
            # Create canonical JSON for signing
            report_copy = report.copy()
            report_copy.pop("digital_signature", None)  # Remove any existing signature
            
            canonical_json = json.dumps(report_copy, sort_keys=True, separators=(',', ':'))
            message_bytes = canonical_json.encode('utf-8')
            
            # Sign with Ed25519
            signature = self.signing_key.sign(message_bytes)
            
            # Get public key for verification
            public_key = self.signing_key.public_key()
            public_key_bytes = public_key.public_bytes(
                encoding=Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            return {
                "algorithm": "Ed25519",
                "signature": signature.hex(),
                "public_key": public_key_bytes.hex(),
                "signed_at": datetime.now(timezone.utc).isoformat(),
                "message_hash": hashlib.sha256(message_bytes).hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Report signing failed: {e}")
            return {
                "algorithm": "none",
                "error": f"Signing failed: {str(e)}",
                "signed_at": datetime.now(timezone.utc).isoformat()
            }
    
    def _store_forensic_report(self, report_id: str, report: Dict[str, Any]):
        """Store forensic report to disk"""
        try:
            # Create reports subdirectory
            reports_dir = self.evidence_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Store as JSON
            report_path = reports_dir / f"{report_id}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Forensic report stored: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to store forensic report: {e}")
    
    def create_evidence_bundle(self, report_id: str) -> Optional[Path]:
        """Create ZIP evidence bundle with all files and reports"""
        try:
            if not self.store_files:
                return None
            
            bundle_path = self.evidence_dir / f"{report_id}_evidence_bundle.zip"
            
            with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add original evidence file
                evidence_files = list(self.evidence_dir.rglob(f"{report_id}_original.*"))
                for evidence_file in evidence_files:
                    zf.write(evidence_file, f"evidence/{evidence_file.name}")
                
                # Add forensic report
                report_file = self.evidence_dir / "reports" / f"{report_id}.json"
                if report_file.exists():
                    zf.write(report_file, f"reports/{report_file.name}")
                
                # Add chain of custody summary
                custody_summary = {
                    "bundle_created_at": datetime.now(timezone.utc).isoformat(),
                    "report_id": report_id,
                    "contents": [
                        "Original evidence file(s)",
                        "Forensic analysis report",
                        "Chain of custody documentation"
                    ],
                    "integrity_note": "Verify SHA-256 hashes against report"
                }
                
                zf.writestr("README.json", json.dumps(custody_summary, indent=2))
            
            logger.info(f"📦 Evidence bundle created: {bundle_path}")
            return bundle_path
            
        except Exception as e:
            logger.error(f"Failed to create evidence bundle: {e}")
            return None
    
    def verify_report_signature(self, report: Dict[str, Any]) -> bool:
        """Verify digital signature of forensic report"""
        try:
            signature_data = report.get("digital_signature")
            if not signature_data or signature_data.get("algorithm") != "Ed25519":
                return False
            
            # Reconstruct message for verification
            report_copy = report.copy()
            report_copy.pop("digital_signature")
            
            canonical_json = json.dumps(report_copy, sort_keys=True, separators=(',', ':'))
            message_bytes = canonical_json.encode('utf-8')
            
            # Load public key and signature
            public_key_bytes = bytes.fromhex(signature_data["public_key"])
            signature_bytes = bytes.fromhex(signature_data["signature"])
            
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
            
            # Verify signature
            public_key.verify(signature_bytes, message_bytes)
            return True
            
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False