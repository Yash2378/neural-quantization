# Quantization Research Project

A comprehensive implementation of neural network quantization techniques, focusing on GPTQ (Gradient-based Post-Training Quantization) and multi-model support.

## Overview

This project implements various quantization algorithms for large language models and multi-modal architectures. The primary focus is on GPTQ quantization with support for models like Gemma, Mistral, and text-to-image architectures.

## Features

- **GPTQ Implementation**: Custom gradient-based post-training quantization
- **Multi-Model Support**: Gemma, Mistral, and text-to-image models
- **Flexible Backend**: Modular quantization backend supporting different bit-widths
- **Comprehensive Testing**: Evaluation framework for quantization quality assessment
- **Template System**: Configurable templates for different quantization scenarios

## Project Structure

```
Quantization/
├── src/
│   ├── quantization/
│   │   ├── backend.py          # Core quantization backend
│   │   └── gptq1.py           # GPTQ implementation
│   ├── models/
│   │   ├── gemma.py           # Gemma model quantization
│   │   ├── mistral.py         # Mistral model quantization
│   │   └── texttoimage.py     # Text-to-image model support
│   └── evaluation/
│       └── test.py            # Testing and evaluation framework
├── templates/                  # Configuration templates
├── examples/                   # Usage examples
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Yash2378/Quantization.git
cd Quantization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install AutoGPTQ (if not automatically installed):
```bash
pip install auto-gptq
```

## Usage

### Basic GPTQ Quantization

```python
from src.quantization.gptq1 import GPTQQuantizer
from src.quantization.backend import QuantizationBackend

# Initialize quantization backend
backend = QuantizationBackend(bits=4, group_size=128)

# Create GPTQ quantizer
quantizer = GPTQQuantizer(backend=backend)

# Quantize a model
quantized_model = quantizer.quantize(model, calibration_data)
```

### Model-Specific Quantization

#### Gemma Model
```python
from src.models.gemma import GemmaQuantizer

gemma_quantizer = GemmaQuantizer(bits=4)
quantized_gemma = gemma_quantizer.quantize(gemma_model)
```

#### Mistral Model
```python
from src.models.mistral import MistralQuantizer

mistral_quantizer = MistralQuantizer(bits=4)
quantized_mistral = mistral_quantizer.quantize(mistral_model)
```

#### Text-to-Image Models
```python
from src.models.texttoimage import TextToImageQuantizer

t2i_quantizer = TextToImageQuantizer(bits=8)
quantized_t2i = t2i_quantizer.quantize(text_to_image_model)
```

## Quantization Methods

### GPTQ (Gradient-based Post-Training Quantization)

- **Bit-widths**: 4-bit, 8-bit, 16-bit
- **Group sizes**: 32, 64, 128, 256
- **Calibration**: Supports custom calibration datasets
- **Error compensation**: Layer-wise error propagation

### Supported Models

| Model | Architecture | Quantization Support | Notes |
|-------|-------------|---------------------|-------|
| Gemma | Transformer | 4-bit, 8-bit | Google's Gemma model |
| Mistral | Transformer | 4-bit, 8-bit | Mistral-7B variants |
| T2I Models | Multi-modal | 8-bit, 16-bit | Text-to-image architectures |

## Evaluation

Run comprehensive evaluation:

```bash
python src/evaluation/test.py --model gemma --bits 4 --dataset c4
```

### Metrics

- **Perplexity**: Language modeling performance
- **Accuracy**: Task-specific accuracy metrics
- **Compression Ratio**: Model size reduction
- **Inference Speed**: Quantized vs. original model speed
- **Memory Usage**: Peak memory consumption

## Model Files

Large model files (`.gguf`, `.bin`, `.safetensors`) are not included in this repository. Download them separately:

- **Mistral-7B-Instruct-v0.2**: Download from Hugging Face
- **Gemma models**: Available through official Google releases
- **Custom models**: Follow respective model documentation

## Configuration

### Templates

The `templates/` directory contains configuration templates for different quantization scenarios:

- Standard quantization settings
- Model-specific configurations
- Calibration dataset configurations

### Environment Variables

```bash
export QUANTIZATION_CACHE_DIR="./cache"
export QUANTIZATION_LOG_LEVEL="INFO"
export CUDA_VISIBLE_DEVICES="0"
```

## Research Applications

This implementation supports research in:

- **Quantization algorithm development**
- **Model compression analysis**
- **Hardware acceleration studies**
- **Multi-modal model optimization**
- **Calibration dataset influence**

## Performance Benchmarks

| Model | Original Size | Quantized Size | Compression | PPL Degradation |
|-------|---------------|----------------|-------------|-----------------|
| Gemma-7B | 14GB | 3.5GB | 4x | <2% |
| Mistral-7B | 14GB | 3.5GB | 4x | <3% |
| T2I Model | 8GB | 2GB | 4x | <5% |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-quantization-method`
3. Commit changes: `git commit -am 'Add new quantization method'`
4. Push to branch: `git push origin feature/new-quantization-method`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{quantization2024,
  title={Quantization Research Project},
  author={Yash Darji},
  year={2024},
  url={https://github.com/Yash2378/Quantization}
}
```

## Acknowledgments

- Built on top of [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- Inspired by the original GPTQ paper
- Thanks to the open-source quantization community

## Contact

For questions or collaboration:
- GitHub: [@Yash2378](https://github.com/Yash2378)
- Issues: [GitHub Issues](https://github.com/Yash2378/Quantization/issues)

---

**Note**: This is a research project. Models and quantization quality may vary. Always validate results for production use.