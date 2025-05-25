
# Soil Image Classification Challenge – 2025  
**Organized by Annam.ai & IIT Ropar**  
**Team Name**: *Ice 'N' Dagger*  
**Members**: Barun Saha, Bibaswan Das  
**Leaderboard Rank**: 70 & 47 

---

##  Project Overview

This repository contains our solutions for both **Phase 1 (Multiclass classification)** and **Phase 2 (Binary anomaly detection)** of the **Soil Image Classification Challenge**. Each phase is organized in its own folder:

```
ANNAM.ai/
├── CHALLENGE-1/    # Phase 1 - Soil Type Classification
└── CHALLENGE-2/    # Phase 2 - Soil vs Non-Soil Detection
```

---

##  Phase 1: Soil Type Classification (Multiclass)

###  Task  
Classify soil images into one of four types:  
**Alluvial**, **Black**, **Clay**, or **Red**

###  Evaluation  
**Minimum per-class F1 score**  
(using `average=None` in `sklearn.metrics.f1_score`)

###  Structure (`CHALLENGE-1/`)
| File               | Purpose |
|--------------------|---------|
| `preprocessing.py` | Load CSVs, encode labels, define dataset class |
| `training.ipynb`   | Train Swin Transformer using `timm` |
| `inference.ipynb`  | Generate predictions using best model |
| `postprocessing.py`| Post-processing and analysis |
| `best_model.pth`   | Saved best model |
| `label_encoder.pkl`| Label encoder for decoding |
| `submission.csv`   | Final predictions |

###  Model  
- **Swin Transformer** (`swin_base_patch4_window7_224`)
- Weighted loss + Stratified split

---

##  Phase 2: Soil vs Non-Soil Classification (Binary)

###  Task  
Identify if a test image is a **soil image (1)** or **not (0)**.  
This is framed as an **anomaly detection** problem since **only positive (soil) examples** are provided for training.

###  Evaluation  
**F1-Score**

###  Structure (`CHALLENGE-2/`)
| File               | Purpose |
|--------------------|---------|
| `preprocessing.py` | Feature extraction using ResNet50 & VGG16 |
| `training.ipynb`   | Train One-Class SVM on soil features |
| `inference.ipynb`  | Test on unknown images using trained model |
| `postprocessing.py`| Format predictions into submission file |
| `decision_scores_analysis.png` | Decision score distribution visualization |

###  Model  
- **Feature Extractors**: ResNet50 + VGG16 (ImageNet-pretrained)
- **Anomaly Detector**: One-Class SVM (`nu=0.01`)

---

##  Setup Instructions

```bash
pip install torch torchvision timm pandas scikit-learn opencv-python tensorflow
```

---

##  Folder Layout

```
ANNAM.ai/
├── CHALLENGE-1/
│   ├── preprocessing.py
│   ├── training.ipynb
│   ├── inference.ipynb
│   ├── postprocessing.py
│   ├── best_model.pth
│   ├── label_encoder.pkl
│   └── submission.csv
│
└── CHALLENGE-2/
    ├── preprocessing.py
    ├── training.ipynb
    ├── inference.ipynb
    ├── postprocessing.py
    └── decision_scores_analysis.png
```

---

##  Submission Instructions

For each phase, upload the following:

### Phase 1
- `preprocessing.py`
- `training.ipynb`
- `inference.ipynb`
- `postprocessing.py`
- `submission.csv` (optional)

### Phase 2
- `preprocessing.py`
- `training.ipynb`
- `inference.ipynb`
- `postprocessing.py`

---

##  Acknowledgements

This work was completed as part of the **Soil Image Classification Challenge 2025** hosted by [Annam.ai](https://annam.ai) and [IIT Ropar](https://www.iitrpr.ac.in).  

We thank the organizers for this valuable opportunity!

## Team Ice 'N' Dagger Members:
- Barun Saha 
- Bibaswan Das
