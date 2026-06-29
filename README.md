# YOLOv11 Solar Panel Detection App

A deep learning computer vision application for detecting and classifying solar panel conditions using the YOLOv11 object detection model. This project demonstrates expertise in building production-ready AI solutions for renewable energy infrastructure monitoring.

## 🎯 Project Overview

This application leverages YOLOv11, a state-of-the-art real-time object detection model, to automatically detect and classify solar panel defects and conditions. The system can identify various panel states including clean panels, dust accumulation, snow coverage, bird droppings, electrical damage, and physical damage—critical for maintenance optimization and efficiency maximization.

## ✨ Key Features

- **Real-time Solar Panel Detection**: YOLOv11-based classification of panel conditions
- **Multi-class Defect Detection**: Identifies 6 panel condition categories:
  - ✅ Clean panels
  - 🌫️ Dusty panels
  - 🐦 Bird-drop damage
  - ⚡ Electrical damage
  - 💔 Physical damage
  - ❄️ Snow-covered panels
- **Comprehensive Notebooks**: End-to-end ML workflow from EDA to deployment
- **Production-Ready Code**: Modular Python implementation for scalable deployment

## 📊 Dataset

This project uses the **Solar Panel Images: Clean and Faulty Dataset** from Kaggle.

**Dataset Citation:**
```bibtex
@dataset{afroz2021solarpanel,
  author = {Afroz, Python},
  title = {Solar Panel Images: Clean and Faulty Images},
  year = {2021},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/pythonafroz/solar-panel-images}},
  note = {Accessed: 2024}
}
```

**Dataset Link:** [pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images)

**Dataset Characteristics:**
- 6 classification categories:
  - Clean - Images of clean solar panels
  - Dusty - Images of dusty solar panels
  - Bird-drop - Images of bird-drop on solar panels
  - Electrical-damage - Images of electrical-damage solar panels
  - Physical-damage - Images of physical-damage solar panels
  - Snow-covered - Images of snow-covered on solar panels
- Web-scraped imagery with natural class imbalance
- Balanced representation across defect types
- Real-world solar panel variations

**Dataset Objective:** Investigate the ability of different machine learning classifiers to detect dust, snow, bird drops, physical and electrical damage on solar panel surfaces with the highest possible accuracy.

**Use Case:** Monitoring and cleaning solar panels is a crucial task to increase modules efficiency, reduce maintenance cost, and reduce the use of resources.

## 🔧 Technical Stack

- **Deep Learning Framework**: PyTorch / YOLOv11
- **Model Architecture**: YOLOv11 (Nano, Small, Medium, Large variants)
- **Development Environment**: Jupyter Notebook (91% of codebase)
- **Production Scripts**: Python (9% of codebase)
- **Image Processing**: OpenCV, PIL/Pillow
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure

```
yolov11-solar-detection-app/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── notebooks/               # Jupyter notebooks (91% of codebase)
│   ├── data_exploration.ipynb        # Dataset analysis & visualization
│   ├── model_training.ipynb          # YOLOv11 fine-tuning
│   ├── inference.ipynb               # Real-time detection demo
│   └── evaluation.ipynb              # Model performance metrics
└── scripts/                 # Python modules (9% of codebase)
    ├── detect.py            # Inference pipeline
    ├── visualize.py         # Detection visualization
    └── utils.py             # Helper functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- 4GB+ RAM (8GB+ recommended)
- GPU highly recommended for training

### Installation

```bash
# Clone repository
git clone https://github.com/nopita3/yolov11-solar-detection-app.git
cd yolov11-solar-detection-app

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Interactive Development (Jupyter Notebooks)
```bash
# Launch Jupyter to explore notebooks
jupyter notebook

# Run through notebooks in order:
# 1. data_exploration.ipynb - Understand dataset
# 2. model_training.ipynb - Train YOLOv11
# 3. inference.ipynb - Test on new images
# 4. evaluation.ipynb - Analyze results
```

#### Production Inference
```bash
# Detect defects in a single image
python scripts/detect.py --image path/to/solar_panel.jpg --model yolov11m

# Batch process multiple images
python scripts/detect.py --image-dir ./images/ --model yolov11m --output results/

# Real-time video detection
python scripts/detect.py --video path/to/video.mp4 --model yolov11m
```

## 🧠 Model Architecture

**YOLOv11 Specifications:**
- **Detection Head**: Multi-scale anchor-free detection
- **Input Resolution**: 640×640 pixels (configurable)
- **Model Variants**: 
  - Nano (fastest) - 4M parameters
  - Small - 11M parameters
  - Medium - 26M parameters (recommended)
  - Large - 44M parameters
- **Output**: Bounding boxes + class probabilities for each detected panel

## 📈 ML Workflow

1. **Exploratory Data Analysis**
   - Dataset statistics and class distribution
   - Image preprocessing and normalization
   - Data augmentation strategies

2. **Model Development**
   - YOLOv11 architecture selection
   - Fine-tuning on solar panel dataset
   - Hyperparameter optimization
   - Cross-validation and evaluation

3. **Inference & Validation**
   - Real-world performance testing
   - Confidence threshold optimization
   - Batch processing capability

4. **Deployment Preparation**
   - Model quantization for edge devices
   - Inference speed optimization
   - Memory-efficient inference

## 🎯 Real-World Applications

### Solar Farm Maintenance
- **Automated Inspection**: Detect defects requiring maintenance
- **Predictive Maintenance**: Schedule cleaning based on dust/snow accumulation
- **Efficiency Monitoring**: Correlate defect detection with energy output

### Property Assessment
- **Solar Potential Evaluation**: Identify installation feasibility
- **Damage Assessment**: Rapid inspection of damage extent
- **Maintenance Cost Estimation**: Prioritize repair resources

### Energy Operations
- **Performance Optimization**: Identify underperforming panels
- **Preventive Maintenance**: Reduce downtime and losses
- **Resource Allocation**: Optimize cleaning schedules and routes

## 📊 Performance Metrics

The model is evaluated on:
- **Accuracy**: Classification accuracy across all defect categories
- **Precision/Recall**: Per-class detection performance
- **mAP (mean Average Precision)**: YOLO's primary metric
- **Inference Speed**: Real-time performance on standard hardware
- **F1-Score**: Balanced performance measurement

## 🔄 Data Processing Pipeline

```
Raw Images → Preprocessing → Augmentation → Training
                                              ↓
                                          YOLOv11
                                              ↓
                          Validation → Optimization → Deployment
```

## 💡 Key Insights & Learnings

- **Class Imbalance Handling**: Techniques for managing web-scraped data imbalance
- **Transfer Learning**: Leveraging pre-trained YOLOv11 weights
- **Real-time Detection**: Balancing accuracy vs. inference speed
- **Domain-Specific Optimization**: Fine-tuning for solar infrastructure

## 📚 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0  # YOLOv11
opencv-python>=4.8.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- [ ] Thermal imaging support
- [ ] Integration with drone flight systems
- [ ] Real-time monitoring dashboard
- [ ] Mobile deployment
- [ ] Additional defect categories

Please submit issues and pull requests!

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Napadol P.** - AI Engineer | Computer Vision | Deep Learning  
Specializing in production ML systems for renewable energy

- 🔗 [GitHub](https://github.com/nopita3)
- 💼 [LinkedIn](https://www.linkedin.com/in/napadol-p-8a000a362)
- 📧 Contact via GitHub profile

## 📞 Support

For questions, issues, or collaboration inquiries:
- 🐛 Open an issue on GitHub
- 📧 Contact via LinkedIn or GitHub

---

**Project Highlights:**
- ✅ End-to-end ML pipeline implementation
- ✅ Production-ready code and documentation
- ✅ Real-world renewable energy application
- ✅ Demonstrated expertise in computer vision and deep learning
- ✅ Scalable architecture for industrial deployment

**Technologies Demonstrated:** PyTorch, YOLOv11, Computer Vision, Transfer Learning, Model Optimization, Data Engineering

## 📖 Citation

If you use this project or the associated dataset in your research, please cite:

**Dataset Citation:**
```bibtex
@dataset{afroz2021solarpanel,
  author = {Afroz, Python},
  title = {Solar Panel Images: Clean and Faulty Images},
  year = {2021},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/pythonafroz/solar-panel-images}}
}
```

**Project Repository:**
```bibtex
@software{napadol2024yolov11solar,
  author = {Napadol P.},
  title = {YOLOv11 Solar Panel Detection App},
  year = {2024},
  url = {https://github.com/nopita3/yolov11-solar-detection-app}
}
```
