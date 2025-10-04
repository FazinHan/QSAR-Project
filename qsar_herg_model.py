#!/usr/bin/env python3
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

Author: QSAR Project Team
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
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

class HERGQSARModel:
    """
    A comprehensive QSAR model for predicting hERG channel blocking activity.
    """
    
    def __init__(self, df, fingerprint_type='morgan', model_type='rf'):
        """
        Initialize the QSAR model.
        
        Parameters:
        -----------
        fingerprint_type : str
            Type of molecular fingerprint ('morgan', 'maccs', 'rdkit', 'combined')
        model_type : str
            Machine learning algorithm ('rf', 'svm', 'xgb' if available)
        """
        self.fingerprint_type = fingerprint_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.X, self.valid_indices = self.prepare_features(df)
        
    def create_sample_dataset(self, n_samples=1000):
        """
        Create a sample dataset with known hERG active/inactive compounds.
        This replaces the ChEMBL API call that was causing issues.
        """
        print("Creating sample dataset for hERG activity prediction...")
        
        # Sample SMILES strings with known hERG activity
        # These are simplified examples - in practice, you'd use a larger curated dataset
        herg_active_smiles = [
            # Known hERG blockers (IC50 < 10 μM)
            'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Chloroquine-like
            'CN(C)CCN1c2ccccc2Sc2ccc(CF3)cc21',   # Trifluoperazine-like
            'COc1cc2c(cc1OC)C(=O)C(CC1CCN(CCc3ccccc3)CC1)C2',  # Verapamil-like
            'Clc1ccc(C(c2ccc(Cl)cc2)N2CCN(CCO)CC2)cc1',  # Cetirizine-like
            'O=C(CCCN1CCC(c2noc3cc(F)ccc23)CC1)c1ccc(F)cc1',  # Risperidone-like
            'CN1CCN(CCCCOC(=O)c2ccc(F)cc2)CC1',   # Haloperidol-like
            'COc1cc(CCN2CCC(O)(c3ccc(Cl)cc3)CC2)ccc1O',  # Loperamide-like
            'CN(C)CCCN1c2ccccc2Sc2ccccc21',       # Promethazine-like
            'CCN1CCCC1CNC(=O)c1cc(N)ccc1OC',      # Procainamide-like
            'CC(C)NCC(O)c1ccc(O)c(CO)c1',         # Albuterol-like
        ]
        
        herg_inactive_smiles = [
            # Known non-hERG blockers (IC50 > 10 μM)
            'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',      # Terbutaline-like
            'CC(=O)Nc1ccc(O)cc1',                  # Acetaminophen
            'CC(C)CC1CCC(C)CC1O',                  # Menthol
            'O=C(O)c1ccccc1O',                     # Salicylic acid
            'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',       # Caffeine
            'CC(C)CCCC(C)C',                       # Isopentane
            'CCO',                                  # Ethanol
            'O',                                    # Water
            'CC(=O)O',                             # Acetic acid
            'C1CCCCC1',                            # Cyclohexane
        ]
        
        # Generate additional diverse molecules using simple transformations
        def generate_variations(base_smiles, n_variations=45):
            """Generate structural variations of base molecules."""
            variations = []
            for smiles in base_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                variations.append(smiles)
                
                # Try simple substitutions and modifications
                for i in range(n_variations // len(base_smiles)):
                    try:
                        # Add methyl groups or modify existing structures slightly
                        modified_smiles = smiles
                        if np.random.random() > 0.5:
                            modified_smiles = smiles.replace('C', 'CC', 1)
                        elif np.random.random() > 0.5:
                            modified_smiles = smiles.replace('c', 'n', 1)
                        
                        test_mol = Chem.MolFromSmiles(modified_smiles)
                        if test_mol is not None and modified_smiles != smiles:
                            variations.append(modified_smiles)
                    except:
                        continue
                        
            return variations[:n_variations]
        
        # Generate more examples
        active_molecules = generate_variations(herg_active_smiles, n_samples // 2)
        inactive_molecules = generate_variations(herg_inactive_smiles, n_samples // 2)
        
        # Create dataframe
        all_smiles = active_molecules + inactive_molecules
        all_labels = [1] * len(active_molecules) + [0] * len(inactive_molecules)
        
        df = pd.DataFrame({
            'canonical_smiles': all_smiles,
            'active': all_labels
        })
        
        # Remove duplicates and invalid SMILES
        df = df.drop_duplicates(subset=['canonical_smiles'])
        valid_smiles = []
        valid_labels = []
        
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            if mol is not None:
                valid_smiles.append(row['canonical_smiles'])
                valid_labels.append(row['active'])
        
        final_df = pd.DataFrame({
            'canonical_smiles': valid_smiles,
            'active': valid_labels
        })
        
        print(f"Created dataset with {len(final_df)} molecules")
        print(f"Active (hERG blockers): {sum(final_df['active'])} molecules")
        print(f"Inactive (non-blockers): {sum(1-final_df['active'])} molecules")
        
        return final_df
    
    def smiles_to_fingerprint(self, smiles, fp_type='morgan'):
        """
        Convert SMILES string to molecular fingerprint.
        
        Parameters:
        -----------
        smiles : str
            SMILES string representation of molecule
        fp_type : str
            Type of fingerprint ('morgan', 'maccs', 'rdkit', 'combined')
            
        Returns:
        --------
        list or None
            Fingerprint as list of bits/features
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            if fp_type == 'morgan':
                # Morgan fingerprint (ECFP4 equivalent)
                fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fp = fpg.GetFingerprint(mol)
                return list(fp)
                
            elif fp_type == 'maccs':
                # MACCS keys (166 bits)
                fp = rdMolDescriptors.GetMACCSKeysDescriptor(mol)
                return list(fp)
                
            elif fp_type == 'rdkit':
                # RDKit fingerprint
                fp = Chem.RDKFingerprint(mol, fpSize=2048)
                return list(fp)
                
            elif fp_type == 'combined':
                # Combine multiple fingerprint types
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                maccs = rdMolDescriptors.GetMACCSKeysDescriptor(mol)
                rdkit_fp = Chem.RDKFingerprint(mol, fpSize=1024)
                
                combined = list(morgan) + list(maccs) + list(rdkit_fp)
                return combined
                
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Prepare molecular features from SMILES strings.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'canonical_smiles' column
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        print(f"Generating {self.fingerprint_type} fingerprints...")
        
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in tqdm(enumerate(df['canonical_smiles']), 
                               total=len(df), desc="Processing molecules"):
            fp = self.smiles_to_fingerprint(smiles, self.fingerprint_type)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        if not fingerprints:
            raise ValueError("No valid fingerprints generated!")
        
        # Create feature matrix
        X = pd.DataFrame(fingerprints)
        
        # Store feature names
        if self.fingerprint_type == 'morgan':
            self.feature_names = [f'Morgan_{i}' for i in range(X.shape[1])]
        elif self.fingerprint_type == 'maccs':
            self.feature_names = [f'MACCS_{i}' for i in range(X.shape[1])]
        elif self.fingerprint_type == 'rdkit':
            self.feature_names = [f'RDKit_{i}' for i in range(X.shape[1])]
        elif self.fingerprint_type == 'combined':
            self.feature_names = ([f'Morgan_{i}' for i in range(1024)] + 
                                [f'MACCS_{i}' for i in range(166)] + 
                                [f'RDKit_{i}' for i in range(1024)])
        
        X.columns = self.feature_names
        
        print(f"Generated feature matrix: {X.shape}")
        return X, valid_indices
    
    def create_model(self, **kwargs):
        """Create the machine learning model based on model_type."""
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'xgb' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def set_model_type(self, model_type):
        """Set the model type and reinitialize the model."""
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        print(f"Model type set to {self.model_type}. Please retrain the model.")
    
    def train(self, df, test_size=0.2, cv_folds=5):
        """
        Train the QSAR model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'canonical_smiles' and 'active' columns
        test_size : float
            Fraction of data to use for testing
        cv_folds : int
            Number of cross-validation folds
        """
        print(f"Training {self.model_type.upper()} model with {self.fingerprint_type} fingerprints...")
        
        # Prepare features
        X, valid_indices = self.X, self.valid_indices
        # X, valid_indices = self.prepare_features(df)
        y = df.iloc[valid_indices]['active'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create and train model with scaling
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.create_model())
        ])
        
        # Fit the model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                  cv=cv_folds, scoring='roc_auc')
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Store results
        self.training_results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cv_scores': cv_scores
        }
        
        self.is_trained = True
        
        # Print results
        print(f"\nCross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Test set AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.training_results
    
    def evaluate_model(self, plot=True):
        """
        Evaluate the trained model and create visualizations.
        
        Parameters:
        -----------
        plot : bool
            Whether to create evaluation plots
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        results = self.training_results
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Model Evaluation Results:")
        print(f"========================")
        print(f"AUC Score: {auc_score:.3f}")
        print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
        print(f"Sensitivity: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")
        print(f"Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]):.3f}")
        
        if plot:
            self.plot_evaluation_results()
        
        return {
            'auc': auc_score,
            'accuracy': (y_pred == y_test).mean(),
            'confusion_matrix': cm,
            'sensitivity': cm[1,1]/(cm[1,1]+cm[1,0]),
            'specificity': cm[0,0]/(cm[0,0]+cm[0,1])
        }
    
    def plot_evaluation_results(self):
        """Create evaluation plots."""
        results = self.training_results
        y_test = results['y_test']
        y_pred_proba = results['y_pred_proba']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        axes[0,0].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,0].set_xlabel('False Positive Rate')
        axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('ROC Curve')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[0,1].plot(recall, precision)
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].set_title('Precision-Recall Curve')
        axes[0,1].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, self.training_results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        axes[1,0].set_title('Confusion Matrix')
        
        # Prediction Distribution
        axes[1,1].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, 
                      label='Non-blockers', color='blue')
        axes[1,1].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, 
                      label='hERG blockers', color='red')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Prediction Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('herg_model_evaluation_{}.png'.format(self.model_type), dpi=300, bbox_inches='tight')
        plt.show()

        print("Evaluation plots saved as 'herg_model_evaluation_{}.png'".format(self.model_type))

    def predict(self, smiles_list):
        """
        Predict hERG blocking activity for new molecules.
        
        Parameters:
        -----------
        smiles_list : list
            List of SMILES strings
            
        Returns:
        --------
        dict
            Dictionary with predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Prepare features
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            fp = self.smiles_to_fingerprint(smiles, self.fingerprint_type)
            if fp is not None:
                fingerprints.append(fp)
                valid_smiles.append(smiles)
            else:
                print(f"Warning: Could not generate fingerprint for {smiles}")
        
        if not fingerprints:
            return {'error': 'No valid molecules to predict'}
        
        # Create feature DataFrame
        X = pd.DataFrame(fingerprints, columns=self.feature_names)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        results = []
        for i, smiles in enumerate(valid_smiles):
            results.append({
                'smiles': smiles,
                'prediction': 'hERG blocker' if predictions[i] == 1 else 'Non-blocker',
                'probability': probabilities[i],
                'risk_level': self._get_risk_level(probabilities[i])
            })
        
        return results
    
    def _get_risk_level(self, probability):
        """Categorize risk level based on probability."""
        if probability < 0.3:
            return 'Low risk'
        elif probability < 0.7:
            return 'Moderate risk'
        else:
            return 'High risk'
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'model': self.model,
            'fingerprint_type': self.fingerprint_type,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.fingerprint_type = model_data['fingerprint_type']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to demonstrate the hERG QSAR model.
    """
    print("hERG Cardiotoxicity QSAR Model")
    print("=" * 50)
    
    # Create model instance
    model = HERGQSARModel(fingerprint_type='morgan', model_type='rf')
    
    # Create sample dataset
    df = model.create_sample_dataset(n_samples=500)
    
    # Train the model
    training_results = model.train(df, test_size=0.2, cv_folds=5)
    
    # Evaluate the model
    evaluation_results = model.evaluate_model(plot=True)
    
    # Example predictions
    test_molecules = [
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Likely hERG blocker
        'CC(=O)Nc1ccc(O)cc1',                  # Likely non-blocker (acetaminophen)
        'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',       # Likely non-blocker (caffeine)
    ]
    
    print("\nExample Predictions:")
    print("-" * 50)
    predictions = model.predict(test_molecules)
    
    for pred in predictions:
        print(f"SMILES: {pred['smiles']}")
        print(f"Prediction: {pred['prediction']}")
        print(f"Probability: {pred['probability']:.3f}")
        print(f"Risk Level: {pred['risk_level']}")
        print("-" * 30)
    
    # Save the model
    model.save_model('herg_qsar_model.pkl')
    
    print("Model training and evaluation complete!")
    print("Check 'herg_model_evaluation.png' for detailed results.")


if __name__ == "__main__":
    main()