# Soil Image Classification Challenge – Phase 1 (Multiclass)

##  Overview
This repository contains the complete pipeline for **Phase 1** of the Soil Classification Challenge organized by Annam.ai & IIT Ropar. The task is to classify soil images into one of four categories: **Alluvial, Black, Clay, or Red** using deep learning.

The evaluation metric is **minimum per-class F1 Score**, emphasizing performance across all classes.

---

##  Project Structure

```
.
├── preprocessing.py       # Data loading, label encoding, and transforms
├── training.ipynb         # Training Swin Transformer model
├── inference.ipynb        # Generating predictions using best saved model
├── postprocessing.py      # Post-processing submission or analysis
└── submission.csv         # Final predictions for submission
```

---

##  Setup

```bash
pip install torch torchvision timm pandas scikit-learn
```

---

##  Pipeline Description

### 1. `preprocessing.py`
- Loads training and test data CSVs.
- Encodes `soil_type` labels.
- Defines a `SoilDataset` class and transformation pipelines.
- Outputs:
  - `encoded_train.csv` with numeric labels.
  - `label_encoder.pkl` for label decoding.

### 2. `training.ipynb`
- Trains a **Swin Transformer** model (`swin_base_patch4_window7_224`) using `timm`.
- Uses stratified split, class weights, and minimum F1 as checkpoint metric.
- Saves the best model to `best_model.pth`.

### 3. `inference.ipynb`
- Loads the best model.
- Applies it to test images to generate predictions.
- Converts predicted labels back to original soil types.
- Saves the result as `submission.csv`.

### 4. `postprocessing.py`
- Loads `submission.csv`.
- (Optional) Analyzes prediction distribution or class-wise statistics.

---

##  Evaluation

The final performance metric is:
```
Minimum F1 Score across all 4 classes
```

Use `sklearn.metrics.f1_score` with `average=None` to get per-class F1 scores.

---

##  Notes

- Ensure all four files are in the same directory before running.
- If running on Kaggle, replace dataset paths accordingly.
- Trained model is saved as `best_model.pth` and reused during inference.

---

##  Submission

Upload the following files:
- `preprocessing.py`
- `training.ipynb`
- `inference.ipynb`
- `postprocessing.py`
- `submission.csv` (optional, for manual inspection)

---

##  Credits

Developed by Team Ice 'N' Dagger as part of **Soil Image Classification Challenge - 2025**, organized by [Annam.ai](https://annam.ai) and [IIT Ropar](https://www.iitrpr.ac.in).

## Team Ice 'N' Dagger Members:
- Barun Saha 
- Bibaswan Das
