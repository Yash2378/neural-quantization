"""
Cascade quantization pipeline: INT8→INT4
Key innovation for achieving 4× compression with <2% degradation
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .gptq_core import ProductionGPTQ
from .backend import QuantizationConfig, QuantizationMethod
import logging

class CascadeQuantizer:
    """
    INT8→INT4 cascade quantization pipeline
    
    Key insight: Gradual precision reduction minimizes accumulated error
    compared to direct FP16→INT4 quantization
    """
    
    def __init__(self, 
                 final_bits: int = 4,
                 intermediate_bits: int = 8,
                 cascade_stages: List[int] = None):
        
        self.final_bits = final_bits
        self.intermediate_bits = intermediate_bits
        
        if cascade_stages is None:
            self.cascade_stages = [intermediate_bits, final_bits]
        else:
            self.cascade_stages = cascade_stages
        
        self.logger = logging.getLogger(__name__)
        
        # Create quantizers for each stage
        self.quantizers = []
        for bits in self.cascade_stages:
            config = QuantizationConfig(
                bits=bits,
                group_size=128,
                symmetric=True,
                error_compensation=True,
                method=QuantizationMethod.GPTQ
            )
            self.quantizers.append(ProductionGPTQ(config))
    
    def quantize_cascade(self, 
                        model: nn.Module,
                        calibration_dataloader: torch.utils.data.DataLoader,
                        target_languages: List[str] = None) -> Dict:
        """
        Perform cascade quantization with cross-lingual validation
        
        Returns comprehensive results including cross-lingual metrics
        """
        self.logger.info(f"Starting cascade quantization: {self.cascade_stages}")
        
        current_model = model
        cascade_results = {
            'stages': [],
            'final_compression_ratio': None,
            'cross_lingual_results': None
        }
        
        # Stage 1: Initial quantization (e.g., FP16 → INT8)
        if len(self.cascade_stages) > 1:
            self.logger.info(f"Stage 1: Quantizing to {self.cascade_stages[0]} bits")
            
            stage1_model, stage1_results = self.quantizers[0].quantize_model(
                current_model, calibration_dataloader
            )
            
            cascade_results['stages'].append({
                'stage': 1,
                'bits': self.cascade_stages[0],
                'results': stage1_results
            })
            
            # Generate intermediate calibration data
            # Key insight: Use quantized model outputs as calibration for next stage
            intermediate_dataloader = self._generate_intermediate_calibration(
                stage1_model, calibration_dataloader
            )
            
            current_model = stage1_model
            calibration_dataloader = intermediate_dataloader
        
        # Final stage: Quantize to target bits (e.g., INT8 → INT4)
        final_stage_idx = len(self.cascade_stages) - 1
        final_bits = self.cascade_stages[final_stage_idx]
        
        self.logger.info(f"Final stage: Quantizing to {final_bits} bits")
        
        final_model, final_results = self.quantizers[final_stage_idx].quantize_model(
            current_model, calibration_dataloader
        )
        
        cascade_results['stages'].append({
            'stage': final_stage_idx + 1,
            'bits': final_bits,
            'results': final_results
        })
        
        # Calculate final compression metrics
        final_compression = self.quantizers[final_stage_idx].backend.calculate_compression_metrics(
            model, final_model
        )
        cascade_results['final_compression_ratio'] = final_compression['compression_ratio']
        cascade_results['memory_reduction_percent'] = final_compression['memory_reduction_percent']
        cascade_results['fits_jetson'] = final_compression['target_fits_jetson']
        
        # Cross-lingual evaluation if requested
        if target_languages:
            self.logger.info("Performing cross-lingual evaluation...")
            cross_lingual_results = self._evaluate_cross_lingual_performance(
                model, final_model, target_languages
            )
            cascade_results['cross_lingual_results'] = cross_lingual_results
        
        self.logger.info("Cascade quantization completed successfully")
        return final_model, cascade_results
    
    def _generate_intermediate_calibration(self,
                                         quantized_model: nn.Module,
                                         original_dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """
        Generate calibration data from intermediate quantized model
        
        This is a key innovation: using quantized model outputs as calibration
        for the next quantization stage preserves learned representations
        """
        self.logger.info("Generating intermediate calibration data...")
        
        intermediate_data = []
        quantized_model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(original_dataloader):
                if batch_idx >= 50:  # Limit for memory efficiency
                    break
                
                # Get model's intermediate representations
                # This captures the quantized model's learned patterns
                device = next(quantized_model.parameters()).device
                
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, 'to') else v 
                            for k, v in batch.items()}
                    try:
                        outputs = quantized_model(**batch)
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            # Use intermediate hidden states
                            intermediate_data.append(outputs.hidden_states[-1].cpu())
                        elif hasattr(outputs, 'last_hidden_state'):
                            intermediate_data.append(outputs.last_hidden_state.cpu())
                    except Exception as e:
                        self.logger.warning(f"Error generating intermediate data: {e}")
                        continue
                else:
                    batch = batch.to(device)
                    try:
                        outputs = quantized_model(batch)
                        if isinstance(outputs, torch.Tensor):
                            intermediate_data.append(outputs.cpu())
                    except Exception as e:
                        self.logger.warning(f"Error generating intermediate data: {e}")
                        continue
        
        # Create new dataloader from intermediate data
        if intermediate_data:
            intermediate_dataset = torch.utils.data.TensorDataset(
                torch.cat(intermediate_data, dim=0)
            )
            intermediate_dataloader = torch.utils.data.DataLoader(
                intermediate_dataset, 
                batch_size=original_dataloader.batch_size,
                shuffle=False
            )
            return intermediate_dataloader
        else:
            self.logger.warning("No intermediate data generated, using original dataloader")
            return original_dataloader
    
    def _evaluate_cross_lingual_performance(self,
                                          original_model: nn.Module,
                                          quantized_model: nn.Module,
                                          target_languages: List[str]) -> Dict:
        """
        Evaluate cross-lingual performance preservation
        
        Critical for ensuring <2% degradation across all languages
        """
        from ..evaluation.multilingual import MultilingualEvaluator
        
        evaluator = MultilingualEvaluator(target_languages)
        
        results = evaluator.evaluate_models(
            original_model=original_model,
            quantized_model=quantized_model
        )
        
        # Check if performance targets are met
        avg_degradation = results.get('average_degradation_percent', 100)
        
        if avg_degradation > 2.0:  # 2% threshold
            self.logger.warning(
                f"Cross-lingual degradation ({avg_degradation:.2f}%) exceeds 2% target"
            )
        else:
            self.logger.info(
                f"Cross-lingual performance target met: {avg_degradation:.2f}% degradation"
            )
        
        return results