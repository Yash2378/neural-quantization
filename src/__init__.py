"""
Neural Quantization Toolkit
Production-grade quantization achieving <2% degradation, 4× compression, 3.2× speedup
"""

from .quantization.backend import QuantizationConfig, KernelBackend
from .models.mistral import MistralQuantizer
from .models.gemma import GemmaQuantizer
from .models.base_quantizer import BaseModelQuantizer
from .evaluation.multilingual import MultilingualEvaluator

__version__ = "1.0.0"
__author__ = "Yash Darji"

__all__ = [
    "QuantizationConfig",
    "KernelBackend", 
    "MistralQuantizer",
    "GemmaQuantizer",
    "BaseModelQuantizer",
    "MultilingualEvaluator",
]