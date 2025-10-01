# hERG Blocker Activity Predictor - Streamlit Web Application

A user-friendly web interface for predicting hERG channel blocking activity (cardiotoxicity) from molecular structures.

## Features

- **SMILES Input**: Enter molecules via SMILES notation in a text box
- **Interactive Drawing**: Draw chemical structures using the integrated Ketcher molecule editor
- **Real-time Predictions**: Instant predictions with probability scores and risk levels
- **2D Visualization**: View the molecular structure of your input
- **Pre-loaded Examples**: Quick-access buttons for testing with known molecules

## Installation

### Prerequisites

The application requires RDKit, which is best installed via conda:

```bash
# Install dependencies using conda (recommended)
conda install -c conda-forge rdkit streamlit scikit-learn xgboost pandas numpy matplotlib seaborn pillow

# Install streamlit-ketcher via pip
pip install streamlit-ketcher
```

Alternatively, if you have RDKit available through other means:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501

## Usage

1. **Enter a Molecule**:
   - Type or paste a SMILES string in the text input box, OR
   - Draw a molecule using the Ketcher editor on the right

2. **View Prediction**:
   - The app will display:
     - Prediction result (hERG blocker or Non-blocker)
     - Probability score (0-100%)
     - Risk level (Low/Moderate/High)
     - 2D molecular structure

3. **Try Examples**:
   - Click the example buttons to test with known molecules:
     - 🔴 Known hERG Blocker (Chloroquine-like)
     - 🟢 Non-Blocker (Acetaminophen)
     - 🟢 Non-Blocker (Caffeine)

## Risk Levels

- 🟢 **Low risk**: Probability < 30%
- 🟡 **Moderate risk**: Probability 30-70%
- 🔴 **High risk**: Probability > 70%

## Technical Details

- **Model**: XGBoost classifier trained on hERG activity data
- **Fingerprint**: Morgan fingerprint (ECFP4 equivalent, 2048 bits)
- **Framework**: Streamlit for the web interface
- **Molecule Drawing**: Ketcher embedded editor
- **Structure Visualization**: RDKit SVG rendering

## Files

- `app.py`: Main Streamlit application
- `requirements.txt`: Python package dependencies
- `herg_qsar_model.pkl`: Pre-trained XGBoost model
- `qsar_herg_model.py`: Model training and prediction code

## Notes

- The model uses Morgan fingerprints calculated from SMILES strings
- Invalid SMILES strings will result in an error message
- The prediction is based on structural features and may not account for all pharmacological factors
