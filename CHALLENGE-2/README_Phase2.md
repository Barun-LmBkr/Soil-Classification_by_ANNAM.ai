# Soil Image Classification Challenge – Phase 2 (Binary Classification)

##  Challenge Organizer:
**Annam.ai**, **IIT Ropar**

##  Team Details:
- **Team Name**: Ice 'N' Dagger  
- **Members**: Barun Saha, Bibaswan Das  
- **Leaderboard Rank**: 70

---

##  Problem Statement

In **Phase 2**, the task is to develop a **binary classification model** that determines whether a given image contains soil (`label = 1`) or not (`label = 0`). Only **positive labeled samples (soil)** are provided in the training set, making this an **anomaly detection** problem.

---

##  Solution Overview

We use a **One-Class SVM** model trained only on soil images. To make this effective:

###  Feature Extraction:
- We extract features using two pre-trained convolutional neural networks:
  - **ResNet50**
  - **VGG16**
- Both models are used **without the top classification layer** and their outputs are concatenated to form rich visual representations.

###  Model Training:
- We fit a **One-Class SVM** on the training features (all soil images).
- A conservative `nu` value is used (`nu=0.01`) to strictly classify only similar images as soil.

###  Prediction:
- During inference, features from test images are extracted using the same pipeline.
- The One-Class SVM predicts if the test image is similar to soil images (`label = 1`) or not (`label = 0`).

---

##  Submission Files

| File              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `preprocessing.py`| Extracts image features using ResNet50 and VGG16 for both training and test |
| `postprocessing.py`| Prepares the final submission file with predictions                        |
| `training.ipynb`  | Trains the One-Class SVM model on extracted features                        |
| `inference.ipynb` | Loads the model, extracts features from test images, and makes predictions  |

---

##  Evaluation Metric

- **F1-Score** (on test set)
- Our approach aims for **high precision** by being conservative in classifying an image as soil.

---

##  Visualizations

- We include histograms and box plots of **decision scores** to understand model confidence.
- These are saved in `decision_scores_analysis.png`.

---

##  Final Notes

- The entire solution is designed with **modularity** and **reproducibility** in mind.
- Feature extraction, model training, and inference are clearly separated for easy debugging and scaling.

---

###  Thank you for the opportunity!

We appreciate the chance to participate and learn from this competition.


## Team Ice 'N' Dagger Members:
- Barun Saha 
- Bibaswan Das