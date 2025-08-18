"""
Base quantizer interface for all model types
Ensures consistent API across Gemma, Mistral, and text-to-image models
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from ..quantization.backend import QuantizationConfig
from ..quantization.cascade import CascadeQuantizer

class BaseModelQuantizer(ABC):
    """
    Abstract base class for model-specific quantizers
    
    Provides consistent interface for:
    - Gemma models
    - Mistral models
    - Text-to-image models
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.cascade_quantizer = CascadeQuantizer(
            final_bits=config.bits,
            cascade_stages=config.cascade_stages
        )
    
    @abstractmethod
    def get_model_architecture_info(self) -> Dict[str, Any]:
        """Return model-specific architecture information"""
        pass
    
    @abstractmethod
    def prepare_calibration_data(self, 
                               raw_data: List[Any],
                               target_languages: List[str] = None) -> torch.utils.data.DataLoader:
        """Prepare model-specific calibration data"""
        pass
    
    @abstractmethod
    def get_quantizable_layers(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Return list of layers that should be quantized"""
        pass
    
    def quantize_model(self, 
                      model: nn.Module,
                      calibration_data: List[Any],
                      target_languages: List[str] = None) -> Tuple[nn.Module, Dict]:
        """
        Standard quantization pipeline for all model types
        
        This method provides consistent quantization across all model architectures
        """
        # Prepare calibration data
        calibration_dataloader = self.prepare_calibration_data(
            calibration_data, target_languages
        )
        
        # Get model architecture info
        arch_info = self.get_model_architecture_info()
        
        # Apply model-specific pre-quantization optimizations
        model = self.pre_quantization_optimization(model)
        
        # Perform cascade quantization
        quantized_model, results = self.cascade_quantizer.quantize_cascade(
            model=model,
            calibration_dataloader=calibration_dataloader,
            target_languages=target_languages or self.config.target_languages
        )
        
        # Apply model-specific post-quantization optimizations
        quantized_model = self.post_quantization_optimization(quantized_model)
        
        # Add architecture info to results
        results['model_architecture'] = arch_info
        results['quantization_config'] = self.config.__dict__
        
        return quantized_model, results
    
    def pre_quantization_optimization(self, model: nn.Module) -> nn.Module:
        """Model-specific optimizations before quantization"""
        return model
    
    def post_quantization_optimization(self, model: nn.Module) -> nn.Module:
        """Model-specific optimizations after quantization"""
        return model
    
    def validate_quantization_quality(self, 
                                    original_model: nn.Module,
                                    quantized_model: nn.Module,
                                    validation_data: List[Any]) -> Dict[str, float]:
        """Validate quantization quality with model-specific metrics"""
        # Default validation - can be overridden by subclasses
        return {
            'compression_ratio': self.cascade_quantizer.quantizers[-1].backend.calculate_compression_metrics(
                original_model, quantized_model
            )['compression_ratio']
        }