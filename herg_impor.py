# QSAR Model for hERG Blockade Prediction
# Part 1: Data Acquisition and Preprocessing

# --- 1. Installation ---
# Make sure you have the necessary libraries installed.
# In a Colab/Jupyter notebook, run these lines in a cell:
# !pip install chembl_webresource_client
# !pip install rdkit-pypi pandas numpy tqdm

# --- 2. Import Libraries ---
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
import numpy as np

# --- 3. Fetch Data from ChEMBL ---
print("Connecting to ChEMBL database...")
# Use the 'new_client' to interact with the ChEMBL API
activity = new_client.activity
target = new_client.target

# Define the ChEMBL ID for the hERG target
HERG_TARGET_ID = 'CHEMBL240'

print(f"Fetching bioactivity data for hERG (Target ID: {HERG_TARGET_ID})...")

# Query for IC50 data for the human hERG protein
res = activity.filter(
    target_chembl_id=HERG_TARGET_ID,
    standard_type="IC50",
    assay_organism="Homo sapiens"
).only(
    'molecule_chembl_id', 'canonical_smiles', 'standard_relation',
    'standard_value', 'standard_units'
)

# Convert the results to a pandas DataFrame
df = pd.DataFrame(res)
print(f"Initial data fetched. Found {len(df)} activity entries.")
print(df.head())

# --- 4. Data Curation and Cleaning ---
print("\n--- Starting Data Curation ---")

# 4.1. Remove entries with missing values
df_clean = df.dropna(subset=['standard_value', 'canonical_smiles'])
print(f"Removed rows with missing values. Shape: {df_clean.shape}")

# 4.2. Filter for standard relation '=' and units 'nM'
# We want exact IC50 values, not ranges like '<' or '>'
df_clean = df_clean[df_clean['standard_relation'] == "'='"]
print(f"Filtered for exact IC50 values ('='). Shape: {df_clean.shape}")

# Ensure all units are in nM for consistency
df_clean = df_clean[df_clean['standard_units'] == 'nM']
print(f"Filtered for standard units 'nM'. Shape: {df_clean.shape}")

# 4.3. Convert 'standard_value' to numeric
# The values are stored as strings, so we need to convert them to numbers
df_clean['standard_value'] = pd.to_numeric(df_clean['standard_value'])

# 4.4. Handle duplicate molecules
# A single compound might be tested multiple times. We'll take the median IC50 value.
# Group by SMILES and calculate the median of the standard_value
df_final = df_clean.groupby('canonical_smiles')['standard_value'].median().reset_index()
print(f"Handled duplicates by taking the median IC50. Shape: {df_final.shape}")


# --- 5. Create Binary Labels ---
print("\n--- Creating Binary Activity Labels ---")
# Define the activity threshold: 10,000 nM (10 µM)
# Compounds with IC50 < 10,000 nM are considered 'active' (blockers)
# Compounds with IC50 >= 10,000 nM are considered 'inactive' (non-blockers)
threshold = 10000.0

df_final['active'] = np.where(df_final['standard_value'] < threshold, 1, 0)
print(f"Activity labels created using a threshold of {threshold} nM.")
print(df_final['active'].value_counts())


# --- 6. Generate Molecular Fingerprints with RDKit ---
print("\n--- Generating Molecular Fingerprints ---")
# Using Morgan Fingerprints (ECFP4 equivalent)
# This function converts a SMILES string to a fingerprint
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Converts SMILES string to a Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Use AllChem.GetMorganFingerprintAsBitVect for a bit vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fp)
    except:
        return None

# Use tqdm to show a progress bar
tqdm.pandas(desc="Generating fingerprints")

# Apply the function to the 'canonical_smiles' column
df_final['fingerprint'] = df_final['canonical_smiles'].progress_apply(smiles_to_fingerprint)

# Drop molecules for which fingerprint generation failed
df_final = df_final.dropna(subset=['fingerprint'])
print(f"Fingerprints generated. Final dataset shape: {df_final.shape}")

# --- 7. Final Processed Data ---
print("\n--- Data Preparation Complete ---")
print("Final DataFrame preview:")
print(df_final.head())

# The 'fingerprint' column contains the features (X) and
# the 'active' column contains the labels (y) for your model.
# You can now proceed to split this data and train your model.

# Example of how to prepare data for scikit-learn
X = pd.DataFrame(df_final['fingerprint'].tolist())
y = df_final['active']

print(f"\nShape of feature matrix (X): {X.shape}")
print(f"Shape of label vector (y): {y.shape}")
