# preprocessing.py
"""

Author: Annam.ai IIT Ropar
Team Name: Ice 'N' Dagger
Team Members: Barun Saha, Bibaswan Das
Leaderboard Rank: 70

"""

# Here you add all the preprocessing related details for the task completed from Kaggle.
import numpy as np
import cv2
import os
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
import warnings
warnings.filterwarnings('ignore')

# Load pre-trained models for feature extraction
def load_feature_extractors():
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    return resnet_model, vgg_model

# Extract features using both ResNet50 and VGG16
def extract_features_from_image(img_path, resnet_model, vgg_model, img_size=(224, 224)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size)

        # Prepare for ResNet50
        img_resnet = np.expand_dims(img_resized, axis=0)
        img_resnet = resnet_preprocess(img_resnet)

        # Prepare for VGG16
        img_vgg = np.expand_dims(img_resized, axis=0)
        img_vgg = vgg_preprocess(img_vgg)

        # Extract features
        resnet_features = resnet_model.predict(img_resnet, verbose=0).flatten()
        vgg_features = vgg_model.predict(img_vgg, verbose=0).flatten()

        # Combine features
        combined_features = np.concatenate([resnet_features, vgg_features])
        return combined_features

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
