# Machine Punishment - Adding Human Judgement to Image Recognition

## Overview
This project introduces an innovative method for incorporating human expertise into Computer Vision models through a saliency-based loss function. Our approach allows humans to guide neural networks by marking important regions in images, improving model generalization while minimizing human effort.

## Key Features
- 🎯 **Saliency Control**: Direct manipulation of model attention through an intuitive interface
- 🧠 **Human-AI Collaboration**: Bridges the gap between human perceptual knowledge and machine learning 
- 📊 **Efficient Learning**: Reduces required training data through expert guidance
- 🔍 **Enhanced Interpretability**: Makes model decisions more transparent and understandable

## How It Works
1. The model processes images using conventional CNN architecture
2. Users can mark regions of interest in images:
   - Green: Areas to emphasize  
   - Red: Areas to de-emphasize
3. A custom loss function incorporates these human judgments into the training process
4. The model adapts its attention mechanisms based on human input

## Applications
- 🏥 Medical Imaging
- 🌍 Geological Exploration  
- 🛰️ Remote Sensing
- 🏭 Manufacturing Quality Control

## Technical Implementation
- Custom loss function for saliency control
- Interactive GUI for image annotation
- Gradient-based optimization techniques
- Validation mechanisms to prevent overfitting

## Results
- Successful control of model attention without performance degradation
- Particularly effective with smaller datasets
- Improved model interpretability
- Enhanced alignment with human expert knowledge

## Future Work
- Comprehensive generalization benchmarking
- Computational efficiency optimization  
- Alternative saliency approaches
- Adaptive saliency optimization

## Getting Started
```python
from MachinePunishment import PunisherLoss

# Initialize the model with PunisherLoss
criterion = PunisherLoss(num_classes, train_dataset, model)
