# 🔍 Deepfake Forensics Framework (DFF)

A comprehensive multi-layer analysis system for deepfake detection and forensic reporting, built with advanced AI models and forensic methodologies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Layers](#framework-layers)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Deepfake Forensics Framework (DFF) is a state-of-the-art system designed to detect and analyze deepfake content through multiple forensic layers. It combines AI-powered detection with traditional forensic techniques to provide comprehensive analysis and expert-level reporting.

### Key Capabilities

- **Multi-Layer Analysis**: 6 specialized forensic layers
- **AI-Powered Detection**: GenConViT model integration
- **Visual Analysis**: Frame-by-frame examination with heatmaps
- **Expert AI Analysis**: LLM-powered forensic assessment
- **Interactive Interface**: Gradio-based web application
- **Comprehensive Reporting**: Detailed forensic reports with recommendations

## ✨ Features

### 🔬 Forensic Analysis Layers

1. **AI Layer** - GenConViT deepfake detection
2. **Metadata Layer** - File information and provenance
3. **Frequency Layer** - DCT/FFT compression artifact analysis
4. **Physics Layer** - Shadow, reflection, and geometry consistency
5. **Contextual Layer** - Reverse image search and social network analysis
6. **Expert Analysis Layer** - LLM-powered forensic assessment

### 🎨 Visual Interface

- **Tabbed Interface**: Organized analysis results
- **Real-time Visualization**: Frame galleries and heatmaps
- **Interactive Reports**: Detailed forensic findings
- **Expert Opinions**: AI-generated professional assessments

### 📊 Advanced Analytics

- **Confidence Scoring**: Multi-layer confidence assessment
- **Risk Assessment**: Automated risk level determination
- **Anomaly Detection**: Frequency and physics-based detection
- **Provenance Tracking**: Source and metadata analysis

## 🏗️ Architecture

```
Deepfake Forensics Framework
├── Core Framework
│   ├── DeepfakeForensicsFramework (orchestrator)
│   └── BaseForensicLayer (abstract base)
├── Analysis Layers
│   ├── AI Layer (GenConViT)
│   ├── Metadata Layer
│   ├── Frequency Layer
│   ├── Physics Layer
│   ├── Contextual Layer
│   └── LLM Analysis Layer
├── Interface
│   └── Gradio Web Application
└── Utilities
    ├── Video Processing
    ├── Image Analysis
    └── Report Generation
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Deepfake_Forensics
   ```

2. **Create conda environment**
   ```bash
   conda create -n DFF python=3.8
   conda activate DFF
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_dff.txt
   ```

4. **Set up Hugging Face token** (for LLM analysis)
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN="your_huggingface_token_here"
   
   # Linux/Mac
   export HF_TOKEN="your_huggingface_token_here"
   ```

### Model Setup

The framework uses the GenConViT model for AI analysis. Ensure the model files are properly placed in the `GenConViT/` directory.

## ⚡ Quick Start

### 1. Launch the Application

```bash
# Using the provided script (Windows)
start_app.bat

# Or manually
python main.py
```

### 2. Access the Interface

Open your browser and navigate to: `http://localhost:7860`

### 3. Analyze a Video

1. Upload a video file using the interface
2. Configure analysis parameters (frames, metadata)
3. Click "🔍 Analyze Video"
4. Review results across all tabs:
   - **📋 Detailed Report**: Comprehensive forensic report
   - **🎨 Visual Analysis**: Frame galleries and visualizations
   - **🤖 Expert Analysis**: AI expert assessment

## 🔬 Framework Layers

### 1. AI Layer (GenConViT)

**Purpose**: Primary deepfake detection using state-of-the-art AI model

**Features**:
- Frame-by-frame analysis
- Confidence scoring
- Fake/Real probability assessment
- Model explainability

**Output**:
```json
{
  "prediction": "FAKE/REAL",
  "confidence": 0.85,
  "fake_probability": 0.85,
  "real_probability": 0.15,
  "model_info": {...}
}
```

### 2. Metadata Layer

**Purpose**: File information and basic forensic analysis

**Features**:
- File size and timestamps
- Creation/modification dates
- Basic integrity checks
- Format validation

### 3. Frequency Layer

**Purpose**: Compression artifact and frequency domain analysis

**Features**:
- DCT (Discrete Cosine Transform) analysis
- FFT (Fast Fourier Transform) analysis
- Compression quality assessment
- Frequency anomaly detection

**Analysis Types**:
- Compression artifacts detection
- Multiple compression identification
- Frequency consistency analysis
- Manipulation pattern detection

### 4. Physics Layer

**Purpose**: Physical consistency and scene analysis

**Features**:
- Shadow consistency analysis
- Reflection analysis
- Geometry and vanishing point analysis
- Object continuity tracking

**Physical Checks**:
- Light source consistency
- Shadow direction and intensity
- Reflection physics
- Geometric perspective
- Object motion continuity

### 5. Contextual Layer

**Purpose**: Provenance and contextual analysis

**Features**:
- Reverse image search
- Social network propagation analysis
- Uploader history and credibility
- Metadata correlation

**Contextual Analysis**:
- Similar image detection
- Platform-specific analysis
- Uploader behavior patterns
- Metadata consistency checks

### 6. Expert Analysis Layer (LLM)

**Purpose**: AI-powered expert forensic assessment

**Features**:
- Professional forensic evaluation
- Multi-layer evidence synthesis
- Risk assessment and recommendations
- Expert-level reporting

**Expert Capabilities**:
- Evidence correlation
- Professional assessment
- Risk level determination
- Actionable recommendations

## 💻 Usage

### Command Line Interface

```python
from dff_framework.core.framework import DeepfakeForensicsFramework
from dff_framework.layers.ai_layer_safe import SafeAILayer

# Initialize framework
framework = DeepfakeForensicsFramework()

# Register layers
ai_layer = SafeAILayer({'net': 'genconvit'})
framework.register_layer("AI_Layer", ai_layer)

# Analyze video
results = framework.analyze_video("path/to/video.mp4")
print(results)
```

### Web Interface

1. **Upload**: Select video file
2. **Configure**: Set analysis parameters
3. **Analyze**: Run comprehensive analysis
4. **Review**: Examine results across tabs
5. **Export**: Save reports and visualizations

### Programmatic Usage

```python
from dff_framework.interface.gradio_app import DFFGradioInterface

# Create interface
app = DFFGradioInterface()

# Launch web interface
app.launch(server_port=7860, share=True)
```

## 📚 API Reference

### Core Framework

#### `DeepfakeForensicsFramework`

Main orchestrator class for the forensic analysis pipeline.

```python
class DeepfakeForensicsFramework:
    def register_layer(self, name: str, layer: BaseForensicLayer)
    def analyze_video(self, video_path: str, options: Dict = None) -> Dict
    def get_results(self) -> Dict
```

#### `BaseForensicLayer`

Abstract base class for all analysis layers.

```python
class BaseForensicLayer(ABC):
    @abstractmethod
    def analyze(self, video_path: str, options: Dict = None) -> Dict
```

### Analysis Layers

#### AI Layer
```python
class SafeAILayer(BaseForensicLayer):
    def __init__(self, config: Dict)
    def analyze(self, video_path: str, options: Dict = None) -> Dict
```

#### Frequency Layer
```python
class FrequencyLayer(BaseForensicLayer):
    def __init__(self, config: Dict)
    def analyze(self, video_path: str, options: Dict = None) -> Dict
```

#### Physics Layer
```python
class PhysicsLayer(BaseForensicLayer):
    def __init__(self, config: Dict)
    def analyze(self, video_path: str, options: Dict = None) -> Dict
```

#### Contextual Layer
```python
class ContextualLayer(BaseForensicLayer):
    def __init__(self, config: Dict)
    def analyze(self, video_path: str, options: Dict = None) -> Dict
```

#### LLM Analysis Layer
```python
class LLMAnalysisLayer(BaseForensicLayer):
    def __init__(self, name: str = "LLM_Analysis_Layer")
    def analyze(self, video_path: str, options: Dict = None) -> Dict
```

## ⚙️ Configuration

### Layer Configuration

#### AI Layer
```python
ai_config = {
    'net': 'genconvit',
    'ed_weight': 'genconvit_ed_inference',
    'vae_weight': 'genconvit_vae_inference',
    'fp16': False,
    'num_frames': 15
}
```

#### Frequency Layer
```python
freq_config = {
    'dct_block_size': 8,
    'fft_threshold': 0.1,
    'compression_quality_range': (30, 100)
}
```

#### Physics Layer
```python
physics_config = {
    'shadow_threshold': 0.3,
    'reflection_threshold': 0.2,
    'geometry_threshold': 0.1,
    'continuity_threshold': 0.4,
    'min_object_area': 1000,
    'max_objects': 10
}
```

#### Contextual Layer
```python
contextual_config = {
    'reverse_search_threshold': 0.8,
    'propagation_threshold': 0.6,
    'uploader_credibility_threshold': 0.5,
    'metadata_correlation_threshold': 0.7
}
```

### Environment Variables

```bash
# Required for LLM analysis
HF_TOKEN="your_huggingface_token"

# Optional for performance
KMP_DUPLICATE_LIB_OK="TRUE"
CUDA_VISIBLE_DEVICES="0"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```
Error: GenConViT model weights not found
```
**Solution**: Ensure model files are in the correct directory structure.

#### 2. CUDA/GPU Issues
```
Error: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU mode.

#### 3. LLM Analysis Failures
```
Error: LLM analysis failed
```
**Solution**: Check HF_TOKEN environment variable.

#### 4. Visual Generation Errors
```
Error: color must be int or single-element tuple
```
**Solution**: Update PIL version or check image processing code.

### Performance Optimization

1. **GPU Memory**: Adjust batch sizes for your GPU
2. **CPU Usage**: Limit concurrent analyses
3. **Storage**: Ensure sufficient disk space for temporary files
4. **Network**: Stable internet for LLM analysis

### Debug Mode

Enable debug mode for detailed logging:

```python
app.launch(debug=True, show_error=True)
```

## 📊 Output Formats

### Analysis Results

The framework generates comprehensive results in multiple formats:

#### JSON Output
```json
{
  "chain_of_custody": {...},
  "analysis_results": {
    "AI_Layer": {...},
    "Metadata_Layer": {...},
    "Frequency_Layer": {...},
    "Physics_Layer": {...},
    "Contextual_Layer": {...},
    "LLM_Analysis_Layer": {...}
  },
  "summary": {...},
  "recommendations": [...]
}
```

#### Visual Outputs
- Frame galleries with annotations
- Heatmaps and visualizations
- Frequency analysis plots
- Physics consistency maps

#### Expert Reports
- Professional forensic assessment
- Risk level determination
- Actionable recommendations
- Evidence correlation

## 📞 Contact

For Contat, colaboration or questions:

- **Email**: yassinechagna01@gmail.com 

---

**Deepfake Forensics Framework v1.0** - Built for the future of digital forensics.
