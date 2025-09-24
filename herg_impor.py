"""
QSAR Model for hERG Blockade Prediction (Cardiotoxicity)
========================================================

A complete machine learning pipeline for predicting hERG channel blocking activity
based on molecular structure. This model classifies molecules as either cardiotoxic
(hERG blockers) or non-cardiotoxic based on their chemical fingerprints.

Features:
- Multiple molecular fingerprint types (Morgan, MACCS, RDKit)
- Various machine learning algorithms (Random Forest, SVM, XGBoost)
- Cross-validation and hyperparameter tuning
- Model evaluation with multiple metrics
- Visualization of results
- Prediction interface for new molecules

Author: fazinhan
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
import os

# Optional: XGBoost (install with: pip install xgboost)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with 'pip install xgboost' for additional model options.")

warnings.filterwarnings('ignore')

