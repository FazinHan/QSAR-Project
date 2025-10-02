#!/usr/bin/env python3
"""
Streamlit Web Application for hERG Blocker Activity Prediction
================================================================

A user-friendly web interface for predicting hERG channel blocking activity
(cardiotoxicity) from molecular structures. Users can input molecules via
SMILES strings or draw them using an interactive molecule editor.

Author: QSAR Project Team
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import base64

try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False
    st.warning("streamlit-ketcher is not available. Molecule drawing feature will be disabled.")


# Page configuration
st.set_page_config(
    page_title="hERG Blocker Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    """Load the pre-trained hERG prediction model."""
    try:
        with open('herg_qsar_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file 'herg_qsar_model.pkl' not found. Please ensure the model file is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def smiles_to_fingerprint(smiles, fp_type='morgan', fpSize=2048):
    """
    Convert SMILES string to molecular fingerprint.
    
    Parameters:
    -----------
    smiles : str
        SMILES string representation of molecule
    fp_type : str
        Type of fingerprint ('morgan', 'maccs', 'rdkit')
    fpSize : int
        Size of the fingerprint
        
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
            fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fpSize)
            fp = fpg.GetFingerprint(mol)
            return list(fp)
            
        elif fp_type == 'maccs':
            # MACCS keys (166 bits)
            fp = rdMolDescriptors.GetMACCSKeysDescriptor(mol)
            return list(fp)
            
        elif fp_type == 'rdkit':
            # RDKit fingerprint
            fp = Chem.RDKFingerprint(mol, fpSize=fpSize)
            return list(fp)
            
    except Exception as e:
        st.error(f"Error processing SMILES {smiles}: {e}")
        return None


def get_risk_level(probability):
    """Categorize risk level based on probability."""
    if probability < 0.3:
        return 'Low risk', '🟢'
    elif probability < 0.7:
        return 'Moderate risk', '🟡'
    else:
        return 'High risk', '🔴'


def draw_molecule(smiles, size=(400, 300)):
    """
    Draw a 2D structure of the molecule from SMILES using SVG.
    
    Parameters:
    -----------
    smiles : str
        SMILES string
    size : tuple
        Image size (width, height)
        
    Returns:
    --------
    str or None
        SVG string of molecule image
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Use SVG drawer which doesn't require Cairo
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        return svg
    except Exception as e:
        st.error(f"Error drawing molecule: {e}")
        return None


def predict_herg_activity(smiles, model_data):
    """
    Predict hERG blocking activity for a given molecule.
    
    Parameters:
    -----------
    smiles : str
        SMILES string of the molecule
    model_data : dict
        Dictionary containing the trained model and metadata
        
    Returns:
    --------
    dict or None
        Prediction results
    """
    if model_data is None:
        return None
    
    # Extract model components
    model = model_data['model']
    fingerprint_type = model_data['fingerprint_type']
    feature_names = model_data['feature_names']
    
    # Generate fingerprint
    fp = smiles_to_fingerprint(smiles, fingerprint_type)
    
    if fp is None:
        st.error("Could not generate molecular fingerprint. Please check your SMILES string.")
        return None
    
    # Create feature DataFrame
    X = pd.DataFrame([fp], columns=feature_names)
    
    # Make prediction
    try:
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        result = {
            'smiles': smiles,
            'prediction': 'hERG blocker' if prediction == 1 else 'Non-blocker',
            'probability': probability,
            'risk_level': get_risk_level(probability)[0],
            'risk_emoji': get_risk_level(probability)[1]
        }
        
        return result
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None


def main():
    """Main application function."""
    
    # Title and description
    st.title("🧬 hERG Blocker Activity Predictor")
    st.markdown("""
    ### Predict Cardiotoxicity Risk
    
    This application predicts whether a molecule is likely to block the hERG potassium channel,
    which is associated with cardiotoxicity. Enter a molecule using either:
    - **SMILES notation** (text input)
    - **Molecular structure drawing** (interactive editor)
    
    ---
    """)
    
    # Load the model
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    # Display model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown(f"""
        - **Model Type:** {model_data.get('model_type', 'Unknown').upper()}
        - **Fingerprint:** {model_data.get('fingerprint_type', 'Unknown').capitalize()}
        - **Status:** {'✅ Trained' if model_data.get('is_trained', False) else '❌ Not Trained'}
        """)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This model predicts hERG channel blocking activity, which is a key indicator 
        of potential cardiotoxicity in drug development.
        
        **Risk Levels:**
        - 🟢 **Low risk:** Probability < 30%
        - 🟡 **Moderate risk:** Probability 30-70%
        - 🔴 **High risk:** Probability > 70%
        """)
    
    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Input (SMILES)")
        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="e.g., CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
            help="Enter a valid SMILES string representing your molecule"
        )
        
        # Example molecules
        st.markdown("**Example molecules:**")
        if st.button("🔴 Known hERG Blocker (Chloroquine-like)"):
            smiles_input = "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"
            st.rerun()
        
        if st.button("🟢 Non-Blocker (Acetaminophen)"):
            smiles_input = "CC(=O)Nc1ccc(O)cc1"
            st.rerun()
        
        if st.button("🟢 Non-Blocker (Caffeine)"):
            smiles_input = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
            st.rerun()
    
    with col2:
        st.subheader("✏️ Draw Molecule")
        if KETCHER_AVAILABLE:
            molecule_drawn = st_ketcher(value="")
            if molecule_drawn:
                st.success("Molecule drawn! Click 'Predict from Drawing' below.")
                if st.button("🔬 Predict from Drawing"):
                    smiles_input = molecule_drawn
        else:
            st.info("Install streamlit-ketcher to enable molecule drawing: `pip install streamlit-ketcher`")
    
    # Prediction section
    if smiles_input and smiles_input.strip():
        st.markdown("---")
        st.header("🔬 Prediction Results")
        
        # Create columns for display
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.subheader("📊 Prediction")
            
            # Make prediction
            with st.spinner("Analyzing molecule..."):
                result = predict_herg_activity(smiles_input.strip(), model_data)
            
            if result:
                # Display results
                st.markdown(f"""
                ### {result['risk_emoji']} {result['prediction']}
                
                - **Risk Level:** {result['risk_level']}
                - **Probability:** {result['probability']:.1%}
                - **SMILES:** `{result['smiles']}`
                """)
                
                # Progress bar for probability
                st.progress(float(result['probability']))
                
                # Additional information
                if result['probability'] > 0.7:
                    st.error("⚠️ High cardiotoxicity risk! This molecule may require structural modification.")
                elif result['probability'] > 0.3:
                    st.warning("⚠️ Moderate risk. Consider further testing or structural optimization.")
                else:
                    st.success("✅ Low cardiotoxicity risk based on current prediction.")
        
        with result_col2:
            st.subheader("🧪 Molecular Structure")
            
            # Draw the molecule
            mol_svg = draw_molecule(smiles_input.strip())
            
            if mol_svg:
                st.image(mol_svg, caption="2D Structure", use_container_width=True)
            else:
                st.error("Could not draw molecule. Please check your SMILES string.")
    else:
        st.info("👆 Enter a SMILES string or draw a molecule to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with Streamlit | Powered by RDKit and scikit-learn | QSAR Project Team
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
