"""
Comprehensive test suite for neural quantization
"""
import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.quantization.backend import QuantizationConfig, ProductionQuantizationBackend
from src.quantization.gptq_core import ProductionGPTQ
from src.quantization.cascade import CascadeQuantizer
from src.models.mistral import MistralQuantizer

class TestQuantizationBackend:
    """Test core quantization backend functionality"""
    
    def test_tensor_quantization(self):
        """Test basic tensor quantization"""
        config = QuantizationConfig(bits=4, group_size=128)
        backend = ProductionQuantizationBackend(config)
        
        # Create test tensor
        test_tensor = torch.randn(1024, 512)
        
        # Quantize
        quantized, scales, zeros = backend.quantize_tensor(test_tensor, bits=4)
        
        # Verify quantization
        assert quantized.dtype == torch.int8
        assert scales.shape[0] == test_tensor.shape[0] // config.group_size
        assert zeros.shape[0] == scales.shape[0]
        
        # Verify quantization bounds
        assert quantized.min() >= -8  # 4-bit signed min
        assert quantized.max() <= 7   # 4-bit signed max
    
    def test_compression_metrics(self):
        """Test compression metrics calculation"""
        config = QuantizationConfig(bits=4)
        backend = ProductionQuantizationBackend(config)
        
        # Create dummy models
        original_model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10))
        quantized_model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10))
        
        metrics = backend.calculate_compression_metrics(original_model, quantized_model)
        
        assert 'compression_ratio' in metrics
        assert 'memory_reduction_percent' in metrics
        assert metrics['compression_ratio'] > 1.0

class TestGPTQCore:
    """Test GPTQ implementation"""
    
    def test_hessian_computation(self):
        """Test Hessian information computation"""
        config = QuantizationConfig(bits=4)
        gptq = ProductionGPTQ(config)
        
        # Create test activations
        activations = torch.randn(100, 64, 512)  # [batch, seq, hidden]
        
        # Compute Hessian
        H = gptq.compute_hessian_information(activations)
        
        assert H.shape == (512, 512)
        assert torch.allclose(H, H.T, atol=1e-6)  # Should be symmetric
        assert torch.all(H.diagonal() > 0)  # Positive definite with damping
    
    @pytest.mark.slow
    def test_layer_quantization(self):
        """Test single layer quantization"""
        config = QuantizationConfig(bits=4, group_size=128)
        gptq = ProductionGPTQ(config)
        
        # Create test layer and activations
        layer = nn.Linear(512, 256)
        activations = torch.randn(1000, 512)
        
        # Quantize layer
        result = gptq.quantize_layer_gptq(layer, activations)
        
        assert 'quantization_error' in result
        assert 'compression_ratio' in result
        assert result['compression_ratio'] == 32 / 4  # FP32 to 4-bit

class TestCascadeQuantization:
    """Test cascade quantization pipeline"""
    
    def test_cascade_stages(self):
        """Test cascade quantization stages"""
        cascade = CascadeQuantizer(final_bits=4, intermediate_bits=8)
        
        assert cascade.cascade_stages == [8, 4]
        assert len(cascade.quantizers) == 2
    
    def test_intermediate_calibration_generation(self):
        """Test intermediate calibration data generation"""
        cascade = CascadeQuantizer()
        
        # Mock model and dataloader
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # Generate intermediate calibration
        intermediate_loader = cascade._generate_intermediate_calibration(model, dataloader)
        
        assert intermediate_loader is not None
        assert len(list(intermediate_loader)) > 0

class TestMistralQuantizer:
    """Test Mistral-specific quantization"""
    
    def test_architecture_info(self):
        """Test Mistral architecture information"""
        config = QuantizationConfig(bits=4)
        quantizer = MistralQuantizer(config)
        
        arch_info = quantizer.get_model_architecture_info()
        
        assert arch_info['model_type'] == 'mistral'
        assert 'quantizable_components' in arch_info
        assert 'q_proj' in arch_info['quantizable_components']
    
    def test_multilingual_calibration(self):
        """Test multilingual calibration data generation"""
        config = QuantizationConfig(bits=4)
        quantizer = MistralQuantizer(config)
        
        base_texts = ["Test text for calibration"]
        target_languages = ['en', 'es', 'fr']
        
        multilingual_samples = quantizer._generate_multilingual_samples(
            base_texts, target_languages
        )
        
        assert len(multilingual_samples) > len(base_texts)
        # Should have samples for each language
        assert any('Explica' in sample for sample in multilingual_samples)  # Spanish
        assert any('Expliquez' in sample for sample in multilingual_samples)  # French

class TestQualityGates:
    """Test production quality gates"""
    
    def test_quality_validation(self):
        """Test quality gate validation"""
        from src.tools.cli import validate_quality_gates
        
        # Mock results that should pass
        good_results = {
            'final_compression_ratio': 4.2,
            'cross_lingual_results': {
                'overall_metrics': {
                    'average_degradation_percent': 1.8,
                    'meets_2_percent_target': True
                },
                'degradation_analysis': {
                    'max_degradation': 2.5
                }
            },
            'fits_jetson': True
        }
        
        validation = validate_quality_gates(good_results, max_degradation=2.0, min_compression=3.5)
        assert validation['passed'] == True
        assert len(validation['failures']) == 0
        
        # Mock results that should fail
        bad_results = {
            'final_compression_ratio': 2.0,  # Too low
            'cross_lingual_results': {
                'overall_metrics': {
                    'average_degradation_percent': 5.0,  # Too high
                    'meets_2_percent_target': False
                }
            },
            'fits_jetson': False
        }
        
        validation = validate_quality_gates(bad_results, max_degradation=2.0, min_compression=3.5)
        assert validation['passed'] == False
        assert len(validation['failures']) > 0

class TestEdgeOptimization:
    """Test edge deployment optimizations"""
    
    def test_jetson_memory_calculation(self):
        """Test Jetson Nano memory calculation"""
        from src.deployment.jetson import JetsonNanoOptimizer
        
        # Test 7B model memory requirements
        memory_analysis = JetsonNanoOptimizer.calculate_memory_requirements(
            model_params=7e9,  # 7B parameters
            sequence_length=2048
        )
        
        assert 'model_memory_gb' in memory_analysis
        assert 'total_memory_gb' in memory_analysis
        assert 'fits_jetson_nano' in memory_analysis
        assert memory_analysis['compression_ratio'] > 1.0

@pytest.mark.integration
class TestEndToEndQuantization:
    """Integration tests for complete quantization pipeline"""
    
    @pytest.mark.slow
    def test_mistral_quantization_pipeline(self):
        """Test complete Mistral quantization pipeline"""
        # This test requires significant memory and time
        # Should be run only in CI/CD environments with sufficient resources
        
        config = QuantizationConfig(
            bits=4,
            group_size=128,
            cascade_stages=[8, 4],
            target_languages=['en', 'es']
        )
        
        quantizer = MistralQuantizer(config)
        
        # Use a small model for testing
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        calibration_data = [
            "This is a test sentence for calibration.",
            "Esta es una oración de prueba para calibración.",
        ] * 10
        
        # Run quantization
        quantized_model, results = quantizer.quantize_model(
            model=model,
            calibration_data=calibration_data,
            target_languages=['en', 'es']
        )
        
        assert quantized_model is not None
        assert 'final_compression_ratio' in results
        assert results['final_compression_ratio'] > 1.0

class TestMultilingualEvaluation:
    """Test multilingual evaluation framework"""
    
    def test_evaluator_initialization(self):
        """Test multilingual evaluator initialization"""
        from src.evaluation.multilingual import MultilingualEvaluator
        
        target_languages = ['en', 'es', 'fr']
        evaluator = MultilingualEvaluator(target_languages)
        
        assert evaluator.target_languages == target_languages
        assert len(evaluator.test_datasets) >= len(target_languages)
    
    def test_text_similarity(self):
        """Test text similarity calculation"""
        from src.evaluation.multilingual import MultilingualEvaluator
        
        evaluator = MultilingualEvaluator(['en'])
        
        # Test identical texts
        similarity = evaluator._calculate_text_similarity("hello world", "hello world")
        assert similarity == 1.0
        
        # Test completely different texts
        similarity = evaluator._calculate_text_similarity("hello world", "goodbye universe")
        assert 0.0 <= similarity <= 1.0
        
        # Test partially similar texts
        similarity = evaluator._calculate_text_similarity("hello world", "hello universe")
        assert 0.0 < similarity < 1.0

# Fixtures for testing
@pytest.fixture
def sample_config():
    """Provide a sample quantization configuration"""
    return QuantizationConfig(
        bits=4,
        group_size=128,
        cascade_stages=[8, 4],
        target_languages=['en', 'es', 'fr']
    )

@pytest.fixture
def sample_model():
    """Provide a sample model for testing"""
    return nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )

@pytest.fixture
def sample_calibration_data():
    """Provide sample calibration data"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Quantization reduces model size while preserving accuracy.",
        "Cross-lingual evaluation ensures fairness across languages.",
        "Edge deployment enables AI on resource-constrained devices."
    ] * 20

# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for quantization operations"""
    
    def test_quantization_speed(self, benchmark, sample_config, sample_model):
        """Benchmark quantization speed"""
        backend = ProductionQuantizationBackend(sample_config)
        test_tensor = torch.randn(1024, 512)
        
        # Benchmark tensor quantization
        result = benchmark(backend.quantize_tensor, test_tensor, 4)
        assert result[0] is not None  # quantized tensor
    
    def test_hessian_computation_speed(self, benchmark, sample_config):
        """Benchmark Hessian computation speed"""
        gptq = ProductionGPTQ(sample_config)
        activations = torch.randn(100, 64, 512)
        
        result = benchmark(gptq.compute_hessian_information, activations)
        assert result.shape == (512, 512)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])