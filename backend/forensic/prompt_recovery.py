"""
Prompt Recovery Module - Metadata-based Exact Prompt Extraction
================================================================
Extracts EXACT prompts from image metadata when available.

This is DIFFERENT from prompt hypothesis/reconstruction:
- Recovery = Exact prompt found in metadata (SIGNED or UNSIGNED)
- Reconstruction = Inferred from visual content (HYPOTHESIS)

Sources checked (in priority order):
1. PNG tEXt/iTXt chunks (Stable Diffusion, ComfyUI, A1111)
2. EXIF UserComment (some generators)
3. XMP:Description (some tools)
4. C2PA manifest (signed provenance)
5. IPTC Caption (rare)

Output status:
- RECOVERED_SIGNED: Prompt from C2PA/signed manifest (highest trust)
- RECOVERED_UNSIGNED: Prompt from metadata (medium trust, can be faked)
- NOT_FOUND: No prompt in metadata

CRITICAL: This module does NOT infer prompts from visual content.
That is handled by prompt_hypothesis.py and prompt_hypothesis_v2.py.
"""

import logging
import re
import json
import struct
import zlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStatus(str, Enum):
    """Status of prompt recovery attempt"""
    RECOVERED_SIGNED = "RECOVERED_SIGNED"      # From C2PA/signed source
    RECOVERED_UNSIGNED = "RECOVERED_UNSIGNED"  # From metadata (can be faked)
    NOT_FOUND = "NOT_FOUND"                    # No prompt in metadata


@dataclass
class RecoveredPrompt:
    """Result of prompt recovery"""
    status: RecoveryStatus = RecoveryStatus.NOT_FOUND
    
    # Recovered prompt data
    prompt: str = ""
    negative_prompt: str = ""
    
    # Generation parameters (if found)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Source information
    source: str = ""  # e.g., "PNG_tEXt", "EXIF_UserComment", "C2PA"
    source_field: str = ""  # Exact field name
    
    # Generator information
    generator: str = ""  # e.g., "Stable Diffusion", "ComfyUI", "Midjourney"
    generator_version: str = ""
    
    # Trust level
    trust_level: str = "none"  # "high" (signed), "medium" (unsigned), "none"
    
    # Raw data for forensic purposes
    raw_data: str = ""
    
    # Warnings/notes
    warnings: List[str] = field(default_factory=list)


class PromptRecovery:
    """
    Extracts exact prompts from image metadata.
    
    Supports:
    - Stable Diffusion (A1111, ComfyUI, Forge, etc.)
    - Midjourney (limited - usually stripped)
    - DALL-E (limited - usually stripped)
    - Other generators that embed prompts
    """
    
    def __init__(self):
        # Known generator signatures
        self.generator_signatures = {
            "automatic1111": ["Steps:", "Sampler:", "CFG scale:"],
            "comfyui": ["workflow", "prompt", "extra_pnginfo"],
            "novelai": ["Source:", "novelai"],
            "invoke": ["invoke", "InvokeAI"],
            "fooocus": ["Fooocus", "fooocus"],
            "forge": ["Forge", "sd-forge"],
            "midjourney": ["Midjourney", "MJ"],
            "dalle": ["DALL-E", "dalle"],
            "stable_diffusion": ["Stable Diffusion", "SD"],
        }
        
        # A1111 parameter patterns
        self.a1111_patterns = {
            "steps": r"Steps:\s*(\d+)",
            "sampler": r"Sampler:\s*([^,]+)",
            "cfg_scale": r"CFG scale:\s*([\d.]+)",
            "seed": r"Seed:\s*(\d+)",
            "size": r"Size:\s*(\d+x\d+)",
            "model": r"Model:\s*([^,]+)",
            "model_hash": r"Model hash:\s*([a-f0-9]+)",
            "clip_skip": r"Clip skip:\s*(\d+)",
            "denoising": r"Denoising strength:\s*([\d.]+)",
            "hires_upscale": r"Hires upscale:\s*([\d.]+)",
            "hires_steps": r"Hires steps:\s*(\d+)",
            "hires_upscaler": r"Hires upscaler:\s*([^,]+)",
            "lora": r"<lora:([^:>]+)(?::[\d.]+)?>"
        }
    
    def recover(self, 
                image_bytes: bytes,
                metadata: Dict[str, Any]) -> RecoveredPrompt:
        """
        Attempt to recover exact prompt from image metadata.
        
        Args:
            image_bytes: Raw image file bytes
            metadata: Extracted metadata dict from MetadataExtractor
        
        Returns:
            RecoveredPrompt with status and data
        """
        result = RecoveredPrompt()
        
        # Try each source in priority order
        
        # 1. Check PNG chunks (highest priority for SD images)
        if self._is_png(image_bytes):
            png_result = self._extract_from_png_chunks(image_bytes)
            if png_result.status != RecoveryStatus.NOT_FOUND:
                return png_result
        
        # 2. Check EXIF UserComment
        exif_result = self._extract_from_exif(metadata)
        if exif_result.status != RecoveryStatus.NOT_FOUND:
            return exif_result
        
        # 3. Check XMP Description
        xmp_result = self._extract_from_xmp(metadata)
        if xmp_result.status != RecoveryStatus.NOT_FOUND:
            return xmp_result
        
        # 4. Check C2PA manifest (if available)
        c2pa_result = self._extract_from_c2pa(metadata)
        if c2pa_result.status != RecoveryStatus.NOT_FOUND:
            return c2pa_result
        
        # 5. Check IPTC Caption
        iptc_result = self._extract_from_iptc(metadata)
        if iptc_result.status != RecoveryStatus.NOT_FOUND:
            return iptc_result
        
        # 6. Check software field for generator hints
        software_result = self._check_software_hints(metadata)
        if software_result.generator:
            result.generator = software_result.generator
            result.warnings.append(f"Generator detected ({software_result.generator}) but no prompt found")
        
        return result
    
    def _is_png(self, data: bytes) -> bool:
        """Check if data is PNG format"""
        return data[:8] == b'\x89PNG\r\n\x1a\n'
    
    def _extract_from_png_chunks(self, data: bytes) -> RecoveredPrompt:
        """
        Extract prompt from PNG tEXt/iTXt/zTXt chunks.
        
        Stable Diffusion and derivatives store prompts in:
        - tEXt chunk with key "parameters" (A1111)
        - tEXt chunk with key "prompt" (ComfyUI)
        - iTXt chunk with key "parameters" or "prompt"
        - zTXt chunk (compressed text)
        """
        result = RecoveredPrompt()
        
        if not self._is_png(data):
            return result
        
        try:
            chunks = self._parse_png_chunks(data)
            
            # Look for text chunks
            for chunk_type, chunk_data in chunks:
                if chunk_type == b'tEXt':
                    key, value = self._parse_text_chunk(chunk_data)
                    prompt_result = self._process_png_text(key, value, "PNG_tEXt")
                    if prompt_result.status != RecoveryStatus.NOT_FOUND:
                        return prompt_result
                
                elif chunk_type == b'iTXt':
                    key, value = self._parse_itxt_chunk(chunk_data)
                    prompt_result = self._process_png_text(key, value, "PNG_iTXt")
                    if prompt_result.status != RecoveryStatus.NOT_FOUND:
                        return prompt_result
                
                elif chunk_type == b'zTXt':
                    key, value = self._parse_ztxt_chunk(chunk_data)
                    prompt_result = self._process_png_text(key, value, "PNG_zTXt")
                    if prompt_result.status != RecoveryStatus.NOT_FOUND:
                        return prompt_result
        
        except Exception as e:
            logger.warning(f"PNG chunk parsing error: {e}")
            result.warnings.append(f"PNG parsing error: {str(e)}")
        
        return result
    
    def _parse_png_chunks(self, data: bytes) -> List[Tuple[bytes, bytes]]:
        """Parse PNG chunks"""
        chunks = []
        pos = 8  # Skip PNG signature
        
        while pos < len(data):
            if pos + 8 > len(data):
                break
            
            length = struct.unpack('>I', data[pos:pos+4])[0]
            chunk_type = data[pos+4:pos+8]
            
            if pos + 12 + length > len(data):
                break
            
            chunk_data = data[pos+8:pos+8+length]
            chunks.append((chunk_type, chunk_data))
            
            pos += 12 + length  # length + type + data + CRC
            
            if chunk_type == b'IEND':
                break
        
        return chunks
    
    def _parse_text_chunk(self, data: bytes) -> Tuple[str, str]:
        """Parse tEXt chunk (key\x00value)"""
        try:
            null_pos = data.index(b'\x00')
            key = data[:null_pos].decode('latin-1')
            value = data[null_pos+1:].decode('latin-1')
            return key, value
        except:
            return "", ""
    
    def _parse_itxt_chunk(self, data: bytes) -> Tuple[str, str]:
        """Parse iTXt chunk (key\x00compression\x00\x00lang\x00transkey\x00text)"""
        try:
            null_pos = data.index(b'\x00')
            key = data[:null_pos].decode('utf-8')
            
            # Skip compression flag, compression method, language tag, translated keyword
            rest = data[null_pos+1:]
            
            # Find the actual text (after 4 null-separated fields)
            parts = rest.split(b'\x00', 4)
            if len(parts) >= 4:
                value = parts[-1].decode('utf-8', errors='replace')
            else:
                value = rest.decode('utf-8', errors='replace')
            
            return key, value
        except:
            return "", ""
    
    def _parse_ztxt_chunk(self, data: bytes) -> Tuple[str, str]:
        """Parse zTXt chunk (key\x00compression_method\x00compressed_text)"""
        try:
            null_pos = data.index(b'\x00')
            key = data[:null_pos].decode('latin-1')
            
            # Skip compression method byte
            compressed = data[null_pos+2:]
            value = zlib.decompress(compressed).decode('latin-1')
            
            return key, value
        except:
            return "", ""
    
    def _process_png_text(self, key: str, value: str, source: str) -> RecoveredPrompt:
        """Process PNG text chunk and extract prompt"""
        result = RecoveredPrompt()
        
        if not key or not value:
            return result
        
        key_lower = key.lower()
        
        # A1111 style "parameters" chunk
        if key_lower == "parameters":
            return self._parse_a1111_parameters(value, source, key)
        
        # ComfyUI style "prompt" chunk (JSON)
        if key_lower == "prompt":
            return self._parse_comfyui_prompt(value, source, key)
        
        # ComfyUI workflow
        if key_lower == "workflow":
            return self._parse_comfyui_workflow(value, source, key)
        
        # NovelAI style
        if key_lower in ["comment", "description", "source"]:
            if any(sig in value for sig in ["Steps:", "Sampler:", "novelai"]):
                return self._parse_a1111_parameters(value, source, key)
        
        return result
    
    def _parse_a1111_parameters(self, value: str, source: str, field: str) -> RecoveredPrompt:
        """
        Parse A1111/Forge style parameters string.
        
        Format:
        prompt text here
        Negative prompt: negative text here
        Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345, Size: 512x512, Model: model_name
        """
        result = RecoveredPrompt()
        result.source = source
        result.source_field = field
        result.raw_data = value
        result.status = RecoveryStatus.RECOVERED_UNSIGNED
        result.trust_level = "medium"
        
        # Split by "Negative prompt:" to get positive and negative
        neg_split = re.split(r'\nNegative prompt:\s*', value, maxsplit=1)
        
        if len(neg_split) >= 1:
            # First part is the positive prompt (may include parameters at end)
            positive_part = neg_split[0].strip()
            
            # Check if parameters are at the end of positive prompt
            param_match = re.search(r'\nSteps:\s*\d+', positive_part)
            if param_match:
                result.prompt = positive_part[:param_match.start()].strip()
                params_str = positive_part[param_match.start():]
            else:
                result.prompt = positive_part
                params_str = ""
        
        if len(neg_split) >= 2:
            negative_part = neg_split[1]
            
            # Negative prompt ends at "Steps:" or similar parameter
            param_match = re.search(r'\nSteps:\s*\d+', negative_part)
            if param_match:
                result.negative_prompt = negative_part[:param_match.start()].strip()
                params_str = negative_part[param_match.start():]
            else:
                # Check for inline parameters
                param_match = re.search(r'Steps:\s*\d+', negative_part)
                if param_match:
                    result.negative_prompt = negative_part[:param_match.start()].strip()
                    params_str = negative_part[param_match.start():]
                else:
                    result.negative_prompt = negative_part.strip()
        
        # Extract parameters
        for param_name, pattern in self.a1111_patterns.items():
            match = re.search(pattern, value)
            if match:
                result.parameters[param_name] = match.group(1)
        
        # Extract LoRAs
        lora_matches = re.findall(self.a1111_patterns["lora"], value)
        if lora_matches:
            result.parameters["loras"] = lora_matches
        
        # Detect generator
        result.generator = self._detect_generator_from_params(value, result.parameters)
        
        logger.info(f"Recovered A1111 prompt: {len(result.prompt)} chars, "
                   f"negative: {len(result.negative_prompt)} chars, "
                   f"params: {len(result.parameters)}")
        
        return result
    
    def _parse_comfyui_prompt(self, value: str, source: str, field: str) -> RecoveredPrompt:
        """Parse ComfyUI JSON prompt"""
        result = RecoveredPrompt()
        result.source = source
        result.source_field = field
        result.raw_data = value[:1000]  # Truncate for storage
        
        try:
            data = json.loads(value)
            
            # ComfyUI stores nodes, find text inputs
            prompts = []
            negatives = []
            
            for node_id, node in data.items():
                if isinstance(node, dict):
                    inputs = node.get("inputs", {})
                    class_type = node.get("class_type", "")
                    
                    # Look for CLIP text encode nodes
                    if "CLIPTextEncode" in class_type:
                        text = inputs.get("text", "")
                        if text:
                            # Heuristic: shorter texts or those with negative words are negatives
                            if any(neg in text.lower() for neg in ["ugly", "deformed", "bad", "worst"]):
                                negatives.append(text)
                            else:
                                prompts.append(text)
                    
                    # Look for KSampler for parameters
                    if "KSampler" in class_type:
                        if "seed" in inputs:
                            result.parameters["seed"] = str(inputs["seed"])
                        if "steps" in inputs:
                            result.parameters["steps"] = str(inputs["steps"])
                        if "cfg" in inputs:
                            result.parameters["cfg_scale"] = str(inputs["cfg"])
                        if "sampler_name" in inputs:
                            result.parameters["sampler"] = inputs["sampler_name"]
            
            if prompts:
                result.prompt = prompts[0]  # Take first/main prompt
                result.status = RecoveryStatus.RECOVERED_UNSIGNED
                result.trust_level = "medium"
                result.generator = "ComfyUI"
            
            if negatives:
                result.negative_prompt = negatives[0]
            
            logger.info(f"Recovered ComfyUI prompt: {len(result.prompt)} chars")
            
        except json.JSONDecodeError:
            result.warnings.append("Invalid JSON in ComfyUI prompt")
        except Exception as e:
            result.warnings.append(f"ComfyUI parsing error: {str(e)}")
        
        return result
    
    def _parse_comfyui_workflow(self, value: str, source: str, field: str) -> RecoveredPrompt:
        """Parse ComfyUI workflow JSON"""
        # Similar to prompt but workflow contains full graph
        return self._parse_comfyui_prompt(value, source, field)
    
    def _detect_generator_from_params(self, raw: str, params: Dict) -> str:
        """Detect generator from parameters and raw text"""
        raw_lower = raw.lower()
        
        # Check for specific signatures
        if "comfyui" in raw_lower or "comfy" in raw_lower:
            return "ComfyUI"
        if "forge" in raw_lower or "sd-forge" in raw_lower:
            return "SD Forge"
        if "automatic1111" in raw_lower or "a1111" in raw_lower:
            return "Automatic1111"
        if "novelai" in raw_lower:
            return "NovelAI"
        if "invoke" in raw_lower:
            return "InvokeAI"
        if "fooocus" in raw_lower:
            return "Fooocus"
        
        # Check model hash format (A1111 specific)
        if params.get("model_hash"):
            return "Stable Diffusion (A1111 family)"
        
        # Generic SD detection
        if params.get("steps") and params.get("sampler"):
            return "Stable Diffusion"
        
        return "Unknown"
    
    def _extract_from_exif(self, metadata: Dict) -> RecoveredPrompt:
        """Extract prompt from EXIF UserComment or ImageDescription"""
        result = RecoveredPrompt()
        
        exif = metadata.get("exif", {})
        
        # Check UserComment
        user_comment = exif.get("UserComment", "")
        if user_comment and len(user_comment) > 20:
            # Check if it looks like a prompt
            if any(sig in user_comment for sig in ["Steps:", "Sampler:", "prompt"]):
                return self._parse_a1111_parameters(user_comment, "EXIF", "UserComment")
        
        # Check ImageDescription
        description = exif.get("ImageDescription", "")
        if description and len(description) > 20:
            if any(sig in description for sig in ["Steps:", "Sampler:"]):
                return self._parse_a1111_parameters(description, "EXIF", "ImageDescription")
        
        return result
    
    def _extract_from_xmp(self, metadata: Dict) -> RecoveredPrompt:
        """Extract prompt from XMP metadata"""
        result = RecoveredPrompt()
        
        xmp = metadata.get("xmp", {})
        
        # Check dc:description
        description = xmp.get("description", "") or xmp.get("dc:description", "")
        if description and len(description) > 20:
            if any(sig in description for sig in ["Steps:", "Sampler:"]):
                return self._parse_a1111_parameters(description, "XMP", "description")
        
        # Check xmp:Label or other fields
        for field in ["Label", "Title", "Subject"]:
            value = xmp.get(field, "")
            if value and "Steps:" in value:
                return self._parse_a1111_parameters(value, "XMP", field)
        
        return result
    
    def _extract_from_c2pa(self, metadata: Dict) -> RecoveredPrompt:
        """
        Extract prompt from C2PA manifest.
        
        C2PA (Content Credentials) provides signed provenance.
        If prompt is in C2PA, it has highest trust level.
        """
        result = RecoveredPrompt()
        
        # C2PA data would be in a special field
        c2pa = metadata.get("c2pa", {}) or metadata.get("content_credentials", {})
        
        if not c2pa:
            return result
        
        # Check for AI generation assertions
        assertions = c2pa.get("assertions", [])
        for assertion in assertions:
            if assertion.get("label") == "c2pa.ai_generated":
                data = assertion.get("data", {})
                
                # Some generators include prompt in C2PA
                prompt = data.get("prompt", "") or data.get("description", "")
                if prompt:
                    result.status = RecoveryStatus.RECOVERED_SIGNED
                    result.trust_level = "high"
                    result.source = "C2PA"
                    result.source_field = "c2pa.ai_generated"
                    result.prompt = prompt
                    result.generator = data.get("generator", "Unknown")
                    
                    # Get parameters if available
                    if "parameters" in data:
                        result.parameters = data["parameters"]
                    
                    logger.info(f"Recovered C2PA signed prompt: {len(result.prompt)} chars")
                    return result
        
        return result
    
    def _extract_from_iptc(self, metadata: Dict) -> RecoveredPrompt:
        """Extract prompt from IPTC Caption/Description"""
        result = RecoveredPrompt()
        
        iptc = metadata.get("iptc", {})
        
        # Check Caption-Abstract
        caption = iptc.get("Caption-Abstract", "") or iptc.get("caption", "")
        if caption and len(caption) > 20:
            if any(sig in caption for sig in ["Steps:", "Sampler:"]):
                return self._parse_a1111_parameters(caption, "IPTC", "Caption-Abstract")
        
        return result
    
    def _check_software_hints(self, metadata: Dict) -> RecoveredPrompt:
        """Check software field for generator hints"""
        result = RecoveredPrompt()
        
        software = metadata.get("software", {})
        software_name = software.get("name", "").lower() if isinstance(software, dict) else ""
        
        # Also check EXIF Software
        exif = metadata.get("exif", {})
        exif_software = exif.get("Software", "").lower()
        
        combined = f"{software_name} {exif_software}"
        
        # Detect generator from software field
        for gen_name, signatures in self.generator_signatures.items():
            if any(sig.lower() in combined for sig in signatures):
                result.generator = gen_name.replace("_", " ").title()
                break
        
        return result


# Global instance
_prompt_recovery: Optional[PromptRecovery] = None

def get_prompt_recovery() -> PromptRecovery:
    global _prompt_recovery
    if _prompt_recovery is None:
        _prompt_recovery = PromptRecovery()
    return _prompt_recovery
