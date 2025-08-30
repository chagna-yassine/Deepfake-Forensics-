# Deepfake Forensics Framework (DfF)

A comprehensive multi-layer analysis system for deepfake detection and forensic reporting, built around the GenConViT model with extensible architecture for additional forensic layers.

## 🎯 Overview

The Deepfake Forensics Framework provides a structured approach to deepfake detection through multiple analysis layers, each contributing to a comprehensive forensic report. The framework is designed to be court-friendly with explainable evidence and reproducible results.

## 🏗️ Architecture

### Current Implementation

- **🤖 AI Layer**: GenConViT model for deepfake detection
- **📋 Metadata Layer**: Basic file information and container analysis

### Planned Layers

- **🔍 Frequency Analysis**: DCT/FFT analysis for compression artifacts
- **🌍 Physics Analysis**: Shadow, reflection, and geometry consistency checks
- **🎯 Localization**: Spatial anomaly detection and heatmaps
- **📊 Explainability**: Attention maps and evidence visualization

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Deepfake_Forensics
```

2. **Install dependencies**:
```bash
pip install -r requirements_dff.txt
```

3. **Download GenConViT model weights**:
```bash
# Download to GenConViT/weight/ directory
wget https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth
wget https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth
```

### Running the Interface

```bash
python main.py
```

The Gradio interface will be available at `http://localhost:7860`

### Command Line Options

```bash
python main.py --help
```

Options:
- `--port`: Port to run the interface on (default: 7860)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--share`: Create a public link
- `--debug`: Enable debug mode

## 📁 Project Structure

```
Deepfake_Forensics/
├── dff_framework/                 # Main framework package
│   ├── core/                     # Core framework components
│   │   ├── framework.py          # Main orchestrator
│   │   └── base_layer.py         # Base class for layers
│   ├── layers/                   # Analysis layers
│   │   ├── ai_layer.py           # GenConViT integration
│   │   └── metadata_layer.py     # Metadata analysis
│   ├── interface/                # User interfaces
│   │   └── gradio_app.py         # Gradio web interface
│   └── utils/                    # Utility functions
├── GenConViT/                    # Original GenConViT implementation
├── main.py                       # Main entry point
├── requirements_dff.txt          # Framework dependencies
└── README_DFF.md                # This file
```

## 🔧 Framework Components

### Core Framework (`dff_framework/core/`)

- **`framework.py`**: Main orchestrator that coordinates all analysis layers
- **`base_layer.py`**: Abstract base class for implementing new analysis layers

### Analysis Layers (`dff_framework/layers/`)

- **`ai_layer.py`**: Integrates GenConViT model for deepfake detection
- **`metadata_layer.py`**: Analyzes video file metadata and container information

### Interface (`dff_framework/interface/`)

- **`gradio_app.py`**: Web-based interface for video analysis

## 🎮 Usage

### Web Interface

1. **Upload Video**: Select a video file for analysis
2. **Configure Analysis**: Set number of frames and analysis options
3. **Run Analysis**: Click "Analyze Video" to start the process
4. **Review Results**: View the comprehensive forensic report

### Programmatic Usage

```python
from dff_framework.core.framework import DeepfakeForensicsFramework
from dff_framework.layers.ai_layer import AILayer
from dff_framework.layers.metadata_layer import MetadataLayer

# Create framework
framework = DeepfakeForensicsFramework()

# Register layers
ai_layer = AILayer({'num_frames': 15})
metadata_layer = MetadataLayer()

framework.register_layer("AI_Layer", ai_layer)
framework.register_layer("Metadata_Layer", metadata_layer)

# Analyze video
results = framework.analyze_video("path/to/video.mp4")
print(results)
```

## 📊 Output Format

The framework generates comprehensive forensic reports including:

### Chain of Custody
- File hash (SHA-256)
- Analysis timestamp
- File metadata

### Analysis Results
- **AI Layer**: Prediction, confidence, probabilities
- **Metadata Layer**: File information, container analysis
- **Additional Layers**: As implemented

### Summary
- Overall confidence score
- Layer success/failure status
- Recommendations

## 🔮 Extending the Framework

### Adding New Analysis Layers

1. **Create a new layer class**:
```python
from dff_framework.core.base_layer import BaseForensicLayer

class MyAnalysisLayer(BaseForensicLayer):
    def __init__(self, config=None):
        super().__init__("My_Analysis", config)
    
    def analyze(self, video_path, options=None):
        # Implement your analysis logic
        return {
            "status": "success",
            "results": "your_analysis_results",
            "confidence": 0.85
        }
```

2. **Register the layer**:
```python
framework.register_layer("My_Analysis", MyAnalysisLayer())
```

### Layer Configuration

Each layer can accept configuration parameters:
```python
layer = MyAnalysisLayer({
    'param1': 'value1',
    'param2': 'value2'
})
```

## 🧪 Testing

Test the framework with sample videos:

```bash
# Run with sample data
python main.py --debug
```

Upload test videos from the `GenConViT/sample_prediction_data/` directory.

## 📈 Performance

- **AI Layer**: ~2-5 seconds per video (depending on length and frames)
- **Metadata Layer**: <1 second per video
- **Memory Usage**: ~2-4GB (depending on model size)

## 🛠️ Development

### Adding New Features

1. **New Analysis Layer**: Implement `BaseForensicLayer`
2. **Enhanced UI**: Modify `gradio_app.py`
3. **Core Improvements**: Update `framework.py`

### Debugging

Enable debug mode for detailed logging:
```bash
python main.py --debug
```

## 📚 References

- **GenConViT Paper**: [Deepfake Video Detection Using Generative Convolutional Vision Transformer](https://arxiv.org/abs/2307.07036)
- **GenConViT Repository**: [GitHub](https://github.com/erprogs/GenConViT)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project builds upon the GenConViT implementation. Please refer to the original GenConViT license for model-related components.

## 🆘 Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error logs and system information

---

**Deepfake Forensics Framework v1.0** - Built for comprehensive deepfake detection and forensic analysis.
