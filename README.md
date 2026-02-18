# 🪙 DeepCoin: Agentic AI for Archaeological Numismatics

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An intelligent multi-agent system that transforms archaeological coin classification from a manual expert task into an automated AI-powered workflow with historical synthesis.**

![DeepCoin Architecture](docs/architecture_diagram.png)

---

## 🎯 Project Overview

**DeepCoin** is a production-grade AI system designed to solve the complex challenge of identifying and classifying ancient coins from photographs. Built for a final-year engineering internship project (PFE) in Tunisia, this project demonstrates cutting-edge integration of:

- 🧠 **Computer Vision**: EfficientNet-B3 CNN for fine-grained image classification
- 🤖 **Agentic AI**: LangGraph-powered multi-agent orchestration
- 📚 **Retrieval-Augmented Generation (RAG)**: Historical context synthesis
- ☁️ **Cloud-Native Architecture**: LocalStack AWS simulation
- 🌐 **Modern Web Stack**: Next.js 15 + FastAPI + PostgreSQL

### The Challenge

Archaeological coins are exceptionally difficult to classify due to:
- **Physical degradation**: Worn by centuries of circulation
- **Corrosion and patina**: Obscured surface details
- **Fragmentation**: Broken or incomplete specimens
- **Data scarcity**: Long-tail distribution with many rare types

### The Solution

A **multi-agent AI system** that:
1. **Enhances** coin images using CLAHE preprocessing
2. **Classifies** using transfer learning (ImageNet → Coins)
3. **Validates** predictions through cross-referencing
4. **Synthesizes** historical reports via RAG + GenAI

---

## ✨ Key Features

### 🔬 Advanced Computer Vision
- **EfficientNet-B3** architecture for optimal accuracy/efficiency
- **CLAHE enhancement** reveals details in worn coins
- **Aspect-preserving preprocessing** prevents geometric distortion
- **Transfer learning** from ImageNet (1.2M images)

### 🤖 Intelligent Agent Orchestration
- **Vision Agent**: CNN-based classification pipeline
- **Research Agent**: RAG system with ChromaDB vector database
- **Validator Agent**: Historical consistency verification
- **Synthesis Agent**: Professional PDF report generation

### 🌐 Production-Ready Architecture
- **FastAPI** backend with async operations
- **Next.js 15** frontend with Server Components
- **PostgreSQL** for persistent storage
- **LocalStack** for AWS S3/Lambda simulation
- **Docker Compose** for orchestration

### 📊 Intelligent Decision Making
```
Confidence > 85%  → Auto-approve & log
Confidence 60-85% → Request human review
Confidence < 60%  → Flag for expert analysis
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  USER INTERFACE (Next.js)                   │
│           Image Upload → Real-time Results → PDF            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND API (FastAPI)                     │
│          /classify | /history | /validate | WebSocket      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR (LangGraph)                       │
│        State Machine with Human-in-the-Loop                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Vision Agent │   │Research Agent│   │Validator Agent│
│ EfficientNet │   │  ChromaDB    │   │ Cross-check  │
│     B3       │   │     RAG      │   │  Historical  │
└──────────────┘   └──────────────┘   └──────────────┘
        ↓                   ↓                   ↓
        └───────────────────┼───────────────────┘
                            ↓
                   ┌──────────────┐
                   │  Synthesis   │
                   │    Agent     │
                   │ PDF Reports  │
                   └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend, coming soon)
- **Docker & Docker Compose**
- **Git**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/deepcoin.git
cd deepcoin

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (CN Dataset v1)
# Place in: data/raw/CN_dataset_v1/

# 5. Run data preprocessing
python src/data_pipeline/prep_engine.py
```

### Project Structure

```
deepcoin/
├── src/
│   ├── data_pipeline/      # Data preprocessing & augmentation
│   │   ├── auditor.py      # Dataset analysis
│   │   └── prep_engine.py  # CLAHE + resizing pipeline
│   ├── core/               # ML core components
│   │   └── model_factory.py # EfficientNet-B3 definition
│   ├── agents/             # LangGraph agent implementations
│   ├── api/                # FastAPI backend
│   └── ...
├── data/
│   ├── raw/                # Original CN dataset (gitignored)
│   ├── processed/          # Preprocessed 299x299 images
│   └── metadata/           # Dataset statistics
├── models/                 # Trained model checkpoints
├── tests/                  # Unit & integration tests
├── notebooks/              # Jupyter notebooks for experiments
└── docker-compose.yml      # Multi-container orchestration
```

---

## 📊 Dataset

### Corpus Nummorum (CN) v1
- **Source**: [corpus-nummorum.eu](https://www.corpus-nummorum.eu/)
- **Total Images**: 115,160 ancient coin photographs
- **Original Classes**: 9,716 unique coin types
- **Challenge**: Severe long-tail distribution

### Our Filtered Dataset
- **Filtered Classes**: 438 coin types (≥10 images each)
- **Total Images**: 7,677 preprocessed images
- **Average per Class**: 17.5 images
- **Format**: 299×299 RGB JPG

**Rationale**: Quality over quantity - CNNs require minimum samples for reliable feature learning.

---

## 🧠 Technical Deep Dive

### Image Preprocessing Pipeline

#### 1. CLAHE Enhancement
```python
# Convert to LAB color space (separates brightness from color)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Apply contrast enhancement to lightness channel only
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)

# Merge back and convert to BGR
img = cv2.merge((l, a, b))
img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
```

**Why LAB?** Enhances contrast without distorting coin colors.

#### 2. Aspect-Preserving Resize
- Maintain original aspect ratio
- Add black padding to reach 299×299
- No geometric distortion of coin features

### Model Architecture

**EfficientNet-B3** (12M parameters)
- **Input**: 299×299×3 RGB images
- **Backbone**: Pre-trained on ImageNet
- **Head**: Custom classifier with dropout (p=0.3)
- **Output**: 438-way softmax (coin types)

**Transfer Learning Strategy**:
1. Load ImageNet weights (generic visual features)
2. Replace classification head
3. Fine-tune entire network on coins
4. Early stopping on validation accuracy

### Multi-Agent System (Coming Soon)

**Vision Agent** → Runs CNN inference  
**Research Agent** → Queries historical database (RAG)  
**Validator Agent** → Cross-references predictions  
**Synthesis Agent** → Generates PDF reports  

**Orchestrator**: LangGraph state machine with conditional routing

---

## 📈 Current Status

### ✅ Completed (Phase 1)
- [x] Project structure & environment setup
- [x] Data auditing & long-tail analysis
- [x] Image preprocessing pipeline (CLAHE + padding)
- [x] EfficientNet-B3 model definition
- [x] Dataset filtering (438 classes, 7,677 images)

### 🔄 In Progress (Phase 2)
- [ ] PyTorch DataLoader with augmentation
- [ ] Training pipeline (loss, optimizer, scheduler)
- [ ] Model evaluation metrics
- [ ] Checkpointing & model versioning

### ⏳ Planned (Phases 3-7)
- [ ] FastAPI backend (/classify endpoint)
- [ ] Next.js 15 frontend
- [ ] LangGraph multi-agent orchestration
- [ ] ChromaDB RAG system
- [ ] LocalStack cloud simulation
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Docker deployment

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 15, TypeScript, Tailwind CSS | Modern web UI |
| **Backend** | FastAPI, Python 3.11, Uvicorn | High-performance API |
| **ML/AI** | PyTorch 2.5, EfficientNet-B3, OpenCV | Computer vision |
| **Agents** | LangGraph, LangChain, OpenAI API | Multi-agent orchestration |
| **Vector DB** | ChromaDB | RAG knowledge base |
| **Database** | PostgreSQL, Redis | Persistence & caching |
| **Cloud Sim** | LocalStack | AWS S3/Lambda emulation |
| **DevOps** | Docker, GitHub Actions, Nginx | CI/CD & deployment |

---

## 📚 Key Concepts

### Long-Tail Distribution
A dataset where few classes have many samples, but most have very few. Our solution: filter classes with <10 images.

### Transfer Learning
Leverage ImageNet pre-training (1.2M images) → fine-tune for coins. **10-100x less data required**.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Local contrast enhancement that reveals worn coin details without amplifying noise.

### Agentic Workflow
Multiple specialized AI agents collaborate (Vision + Research + Validator) instead of a single monolithic model.

### RAG (Retrieval-Augmented Generation)
Combine LLM generation with real-time document retrieval to prevent hallucinations.

---

## 🎓 Academic Context

**Institution**: [Your University], Tunisia  
**Project Type**: Final Year Engineering Internship (PFE)  
**Domain**: Computer Vision, Deep Learning, Agentic AI  
**Duration**: 16 weeks (February - June 2026)  

**Supervisor Requirements Met**:
- ✅ Archaeological coin recognition system
- ✅ CNN implementation (EfficientNet)
- ✅ Image preprocessing pipeline
- ✅ Database construction & filtering

**Value-Added Beyond Requirements**:
- 🚀 Multi-agent orchestration (LangGraph)
- 🚀 RAG historical synthesis
- 🚀 Cloud-native architecture (LocalStack)
- 🚀 Production-ready deployment

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for Contribution**:
- Data augmentation strategies
- Model architecture experiments
- Agent prompt engineering
- Frontend UI/UX improvements

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Corpus Nummorum (CN)** for providing the archaeological dataset
- **PyTorch** and **torchvision** teams for excellent ML frameworks
- **LangChain** team for LangGraph agent orchestration tools
- **FastAPI** creators for the modern Python web framework

---

## 📞 Contact

**Author**: [Your Name]  
**Email**: [your.email@example.com]  
**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
**GitHub**: [@yourusername](https://github.com/yourusername)  

**Project Link**: [https://github.com/yourusername/deepcoin](https://github.com/yourusername/deepcoin)

---

## 🏆 Project Goals

### Technical Excellence
- **Accuracy**: Target >85% on test set
- **Speed**: <500ms inference time per image
- **Scalability**: Handle 1000+ concurrent requests (simulated)

### Career Impact
- **Portfolio Differentiator**: Stand out in fullstack/AI job applications
- **Technical Depth**: Demonstrate end-to-end system design skills
- **Modern Stack**: Showcase 2026 industry-standard technologies

### Academic Achievement
- **PFE Grade**: Excellent (18+/20)
- **Innovation**: Multi-agent approach exceeds typical student projects
- **Documentation**: Comprehensive technical writing

---

<div align="center">

**⭐ Star this project if you find it interesting!**

Built with ❤️ for archaeological preservation and AI innovation

</div>
