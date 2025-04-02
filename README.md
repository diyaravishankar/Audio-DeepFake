## Audio Deepfake Detection and Analysis 

Welcome to the Audio Deepfake Detection and Analysis repository! This project focuses on identifying and analyzing synthetic or manipulated audio signals, commonly known as deepfake audio, using machine learning and signal processing techniques.

Here we will be reviewing 3 end to end models for this usecase: 
  - ✅ [AASIST (Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks)](https://github.com/clovaai/aasist)
  - ✅ [Light-DARTS](https://arxiv.org/pdf/2208.09618)
  - ✅ [ECAPA-TDNN + MFCC](https://arxiv.org/pdf/2210.17222)

## 🧠 Technical Comparison

### Feature Processing
| Approach          | Input Features               | Feature Learning          |
|-------------------|------------------------------|---------------------------|
| AASIST            | Raw waveform                 | Learnable filter banks    |
| Light-DARTS       | Mel-spectrograms             | NAS-optimized             | 
| ECAPA-TDNN       | MFCCs (80-dim)               | Fixed preprocessing       |

### AASIST (Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks)
**Key Innovation**:  
- ✅ Integrated spectro-temporal graph attention networks for cross-domain feature interaction
- ✅ Dual-branch architecture with learnable filter banks
- ✅ Heterogeneous graph stacking for spoofing artifact detection

**Performance**:  
- **ASVspoof 2019 LA**: 0.83% EER 
- **Cross-Dataset**: 5.21% EER
- **In-the-Wild**: 34.81% EER

**Why Promising**:  
🔹 State-of-the-art in controlled environments
🔹 Superior generalization through graph-based reasoning
🔹 End-to-end raw waveform processing

**Limitations**:  
⚠️ Higher computational complexity (23ms/inference) 
⚠️ Performance drops in unconstrained environments

---

### Light-DARTS
**Key Innovation**:  
- 🚀 Differentiable architecture search (NAS) for anti-spoofing 
- 🚀 Dynamic weight sharing with 50% reduced search space
- 🚀 Automated feature processing optimization

**Performance**:  
- **ASVspoof 2019 LA**: 1.12% EER
- **Training Efficiency**: 50hrs search + 20hrs training 

**Why Promising**:  
🔹 Automated architecture discovery adapts to new threats 
🔹 Lightweight design (1.2M params) enables deployment 

**Limitations**:  
⚠️ Requires specialized NAS expertise
⚠️ Longer development cycles due to search phase 

---

### ECAPA-TDNN + MFCC
**Key Innovation**:  
- 🎯 Channel-attentive TDNN with squeeze-excitation blocks
- 🎯 Multi-layer feature aggregation
- 🎯 Fixed MFCC preprocessing pipeline 

**Performance**:  
- **ASVspoof 2019 LA**: 1.45% EER 
- **ADD2023 Challenge**: 75.41% F1-score  
- **Inference Speed**: 8ms/utterance 

**Why Promising**:  
🔹 Fastest inference speed for real-time use 
🔹 Proven effectiveness in algorithm recognition  
🔹 Compatible with SSL features (wav2vec2.0)

**Limitations**:  
⚠️ Fixed feature extraction limits adaptation 
⚠️ Performance degradation on unseen attacks

---


## ⚙️ Implementation Details

### Key Specifications
| Metric             | AASIST     | Light-DARTS | ECAPA-TDNN |
|--------------------|------------|-------------|------------|
| Parameters         | 2.8M       | 1.2M        | 4.3M       |
| Training Time*     | 18hrs      | 50hrs+20hrs | 12hrs      |
| Inference Speed    | 23ms       | 15ms        | 8ms        |

*On 4x V100 GPUs with ASVspoof 2019 LA dataset

## 🚀 Usage Examples

### AASIST Inference
python 
```
detect.py --model aasist
--input sample.wav
--checkpoint weights/aasist.pth
```

### Light-DARTS Search

```
from light_darts import SearchController

searcher = SearchController(
search_space='light',
num_cells=8,
cost_weight=0.5
)
searcher.train_search(dataset)
```

### ECAPA-TDNN Speaker Embedding Extraction

```
import torch
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

# Define input features (batch_size, seq_len, feature_dim)
input_feats = torch.rand([5, 120, 80])  # Example input tensor

# Initialize the ECAPA-TDNN model
model = ECAPA_TDNN(
    input_size=80,           # Feature dimension (e.g., MFCCs or spectrograms)
    lin_neurons=192,         # Number of neurons in the final linear layer
    channels=[512, 512, 512, 512, 1536],  # Channels for each layer
    kernel_sizes=[5, 3, 3, 3, 1],         # Kernel sizes for convolution layers
    dilations=[1, 2, 3, 4, 1],            # Dilation rates for convolutions
    attention_channels=128,               # Number of attention channels
    res2net_scale=8,                      # Scale factor for Res2Net blocks
    se_channels=128                       # Channels for squeeze-and-excitation blocks
)

# Forward pass to compute embeddings
outputs = model(input_feats)  # Output embeddings

# Print output shape (batch_size, embedding_dim)
print(outputs.shape)  # Expected shape: [5, 1, 192]
```


## 📊 Performance Benchmarks

| Dataset            | AASIST (EER%) | Light-DARTS (EER%) | ECAPA-TDNN (EER%) |
|--------------------|---------------|--------------------|-------------------|
| ASVspoof 2019 LA   | 0.83          | 1.12               | 1.45              |
| Deepfake-TIMIT     | 2.15          | 3.01               | 4.32              |
| Cross-Dataset (A→B)| 5.21          | 7.89               | 9.45              |


## 🏆 Model Pick: AASIST

  - EER: 0.44%

  - Dataset: ASVspoof 2019 (LA)

  - Feature Extraction: Wav2Vec2, HuBERT, WavLM

  - Network: Conv-TasNet

  - Loss Function: AAM Softmax

## 📈Performance:

  - Achieves 14.07% EER on ASVSpoof2019 LA music-0dB condition

  - Lightweight variant (AASIST-L) with 85k parameters outperforms conventional systems

  - Maintains 22.67% EER in closed-set synthetic audio detection

## 💪Strengths:

  - Unified architecture eliminates need for score-level ensembles

  - Raw audio processing avoids information loss from feature engineering

  - Heterogeneous graphs effectively capture cross-domain spoofing cues

## ⚠️ Limitations:

  - Higher computational complexity from graph networks

  - Requires careful tuning of attention mechanisms


## 📚 References
- AASIST: [Official GitHub](https://github.com/clovaai/aasist)
- Light-DARTS: [Implementation Guide](https://github.com/light-darts/docs)
- ECAPA-TDNN: [Original Paper](https://arxiv.org/pdf/2210.17222)


