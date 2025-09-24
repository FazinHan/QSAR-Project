#!/usr/bin/env python3
"""
Example Usage of hERG QSAR Model
=================================

This script demonstrates how to use the hERG cardiotoxicity prediction model
for different scenarios:

1. Training a new model with custom parameters
2. Loading a pre-trained model
3. Making predictions on new molecules
4. Comparing different model types and fingerprints

"""

from qsar_herg_model import HERGQSARModel
import pandas as pd

def compare_models():
    """Compare different model types and fingerprint combinations."""
    print("Comparing Different Model Configurations")
    print("=" * 50)
    
    # Model configurations to test
    configs = [
        ('morgan', 'rf'),
        ('maccs', 'rf'),
        ('morgan', 'svm'),
        ('combined', 'rf'),
    ]
    
    # Add XGBoost if available
    try:
        import xgboost
        configs.append(('morgan', 'xgb'))
    except ImportError:
        pass
    
    results = []
    
    for fp_type, model_type in configs:
        print(f"\nTesting {model_type.upper()} with {fp_type} fingerprints...")
        
        # Create model
        model = HERGQSARModel(fingerprint_type=fp_type, model_type=model_type)
        
        # Create dataset
        df = model.create_sample_dataset(n_samples=200)
        
        # Train model
        model.train(df, test_size=0.3, cv_folds=3)
        
        # Evaluate
        eval_results = model.evaluate_model(plot=False)
        
        results.append({
            'fingerprint': fp_type,
            'model': model_type,
            'auc': eval_results['auc'],
            'accuracy': eval_results['accuracy'],
            'sensitivity': eval_results['sensitivity'],
            'specificity': eval_results['specificity']
        })
    
    # Print comparison table
    print("\nModel Comparison Results:")
    print("-" * 80)
    print(f"{'Fingerprint':<12} {'Model':<8} {'AUC':<6} {'Acc':<6} {'Sens':<6} {'Spec':<6}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['fingerprint']:<12} {result['model']:<8} "
              f"{result['auc']:.3f}  {result['accuracy']:.3f}  "
              f"{result['sensitivity']:.3f}  {result['specificity']:.3f}")


def predict_drug_molecules():
    """Test predictions on some real drug molecules."""
    print("\nPredicting Cardiotoxicity for Known Drugs")
    print("=" * 50)
    
    # Load pre-trained model (or create new one if not available)
    try:
        model = HERGQSARModel()
        model.load_model('herg_qsar_model.pkl')
        print("Loaded pre-trained model")
    except FileNotFoundError:
        print("Creating new model...")
        model = HERGQSARModel(fingerprint_type='morgan', model_type='rf')
        df = model.create_sample_dataset(n_samples=500)
        model.train(df)
        model.save_model('herg_qsar_model.pkl')
    
    # Test molecules (real drug structures)
    test_drugs = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Ibuprofen': 'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
        'Paracetamol': 'CC(=O)Nc1ccc(O)cc1',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
        'Diphenhydramine': 'CN(C)CCOC(c1ccccc1)c2ccccc2',
        'Loratadine': 'CCOC(=O)N1CCC(=C2c3ccccc3CCc4ccccc42)CC1',
        'Terfenadine': 'CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccc(C(F)(F)F)cc3)c4ccccc4)CC2)cc1',
        'Cisapride': 'COc1cc(N)c(Cl)cc1C(=O)N2CCC(CC2)C(=O)N3CCN(CCF)CC3'
    }
    
    # Make predictions
    smiles_list = list(test_drugs.values())
    predictions = model.predict(smiles_list)
    
    # Display results
    drug_names = list(test_drugs.keys())
    print(f"{'Drug Name':<15} {'Prediction':<15} {'Probability':<12} {'Risk Level':<12}")
    print("-" * 60)
    
    for i, pred in enumerate(predictions):
        print(f"{drug_names[i]:<15} {pred['prediction']:<15} "
              f"{pred['probability']:<12.3f} {pred['risk_level']:<12}")


def batch_screening():
    """Demonstrate batch screening of a compound library."""
    print("\nBatch Screening Example")
    print("=" * 30)
    
    # Create model
    model = HERGQSARModel(fingerprint_type='morgan', model_type='rf')
    df = model.create_sample_dataset(n_samples=300)
    model.train(df, test_size=0.2)
    
    # Simulate a compound library
    compound_library = [
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Chloroquine-like
        'CC(=O)Nc1ccc(O)cc1',                  # Acetaminophen
        'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',       # Caffeine
        'CCO',                                  # Ethanol
        'COc1cc2c(cc1OC)C(=O)CC2',            # Simple structure
        'c1ccc2c(c1)ccc3c2ccc4c3cccc4',       # Pyrene
        'CC(C)CC1CCC(C)CC1O',                 # Menthol
        'O=C(O)c1ccccc1O',                     # Salicylic acid
    ]
    
    # Screen the library
    print("Screening compound library...")
    predictions = model.predict(compound_library)
    
    # Categorize results
    high_risk = [p for p in predictions if p['risk_level'] == 'High risk']
    moderate_risk = [p for p in predictions if p['risk_level'] == 'Moderate risk']
    low_risk = [p for p in predictions if p['risk_level'] == 'Low risk']
    
    print(f"\nScreening Results:")
    print(f"Total compounds screened: {len(predictions)}")
    print(f"High risk (cardiotoxic): {len(high_risk)}")
    print(f"Moderate risk: {len(moderate_risk)}")
    print(f"Low risk (safe): {len(low_risk)}")
    
    if high_risk:
        print("\nHigh Risk Compounds:")
        for compound in high_risk:
            print(f"  SMILES: {compound['smiles']}")
            print(f"  Probability: {compound['probability']:.3f}")


def custom_training_example():
    """Example of training with custom parameters."""
    print("\nCustom Training Example")
    print("=" * 30)
    
    # Create model with custom parameters
    model = HERGQSARModel(fingerprint_type='combined', model_type='rf')
    
    # Generate larger dataset
    df = model.create_sample_dataset(n_samples=800)
    
    # Train with custom parameters
    print("Training with custom parameters...")
    model.train(df, test_size=0.25, cv_folds=10)
    
    # Detailed evaluation
    eval_results = model.evaluate_model(plot=True)
    
    print(f"\nDetailed Results:")
    print(f"AUC Score: {eval_results['auc']:.4f}")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Sensitivity (True Positive Rate): {eval_results['sensitivity']:.4f}")
    print(f"Specificity (True Negative Rate): {eval_results['specificity']:.4f}")
    
    # Save the model
    model.save_model('custom_herg_model.pkl')
    print("Custom model saved as 'custom_herg_model.pkl'")


def main():
    """Main function to run all examples."""
    print("hERG QSAR Model - Usage Examples")
    print("=" * 40)
    
    # Example 1: Compare different models
    compare_models()
    
    # Example 2: Predict known drugs
    predict_drug_molecules()
    
    # Example 3: Batch screening
    batch_screening()
    
    # Example 4: Custom training
    custom_training_example()
    
    print("\n" + "=" * 40)
    print("All examples completed successfully!")
    print("Check the generated plots and model files.")


if __name__ == "__main__":
    main()