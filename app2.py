import streamlit as st
import os
import openai
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from PIL import Image
import io
import re
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import torch
import pandas as pd
import numpy as np
import json
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

# ===========================
# 0. Configure Logging
# ===========================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===========================
# 1. Streamlit Interface Setup
# ===========================

st.set_page_config(page_title="🧪 Smart Reaction Design Tool", layout="wide")

st.title("🧪 Smart Reaction Design Tool")

# Sidebar for API Key input
st.sidebar.header("🔑 Configuration")
api_key = st.sidebar.text_input("Enter your Sambanova API Key:", type="password")

if not api_key:
    st.sidebar.warning("Please enter your Sambanova API Key to proceed.")
    st.stop()

# ===========================
# 2. Configure Sambanova API
# ===========================

# Configure OpenAI to use Sambanova's API
openai.api_key = api_key
openai.api_base = "https://api.sambanova.ai/v1"  # Sambanova's API base URL
openai.api_type = "open_ai"  # Assuming Sambanova follows OpenAI's API structure
openai.api_version = "2023-03-15-preview"  # Replace with the correct version if different

# ===========================
# 3. Define Utility Functions
# ===========================

def is_valid_smiles(smiles, max_length=200):
    """
    Check if the SMILES string is valid and not too long.
    Args:
        smiles (str): The SMILES string to validate.
        max_length (int): Maximum allowed length of the SMILES string.
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return len(smiles) <= max_length
    except:
        return False

def draw_molecule(smiles, filename=None):
    """
    Generate an image of the molecule from its SMILES string.
    Args:
        smiles (str): The SMILES string of the molecule.
        filename (str, optional): If provided, saves the image to the given path.
    Returns:
        PIL.Image or None: Image of the molecule or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        if filename:
            img.save(filename)
            return None
        return img
    else:
        logger.warning(f"Could not parse molecule: {smiles}")
        return None

def extract_smiles(input_str):
    """
    Extract SMILES strings from an input string using regex.
    Args:
        input_str (str): The input string containing SMILES.
    Returns:
        list: List of extracted SMILES strings.
    """
    smiles = re.findall(r'([A-Z][A-Za-z0-9@+\-\[\]\(\)\\\/#\.=]+)', input_str)
    return smiles

# ===========================
# 4. Define Custom Dataset for Fine-Tuning
# ===========================

class MolecularDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# ===========================
# 5. Define Custom Regression Model
# ===========================

from torch import nn  # Ensure that 'nn' is imported

class ChemBERTaForMultiOutputRegression(nn.Module):
    def __init__(self, model_name, hidden_size=768, num_labels=2):
        super(ChemBERTaForMultiOutputRegression, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(hidden_size, num_labels)  # Two outputs: LogP and LogS

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        predictions = self.regressor(cls_embedding)
        
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)
            return loss, predictions
        else:
            return predictions

# ===========================
# 6. Load Models
# ===========================

@st.cache_resource
def load_chemberta_model(model_name="seyonec/ChemBERTa-zinc-base-v1"):
    """
    Load the ChemBERTa model and tokenizer for fine-tuning.
    Returns:
        tokenizer: The tokenizer for ChemBERTa.
        model: The ChemBERTa model with a regression head.
        device: The device the model is loaded onto.
    """
    try:
        logger.info("Loading ChemBERTa tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ChemBERTaForMultiOutputRegression(model_name=model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info("ChemBERTa model loaded successfully.")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Error loading ChemBERTa model and tokenizer: {e}")
        st.error(f"Error loading ChemBERTa model and tokenizer: {e}")
        st.stop()

@st.cache_resource
def load_chemgpt_model():
    """
    Load the ChemGPT model and tokenizer from Hugging Face.
    Returns:
        tokenizer: The tokenizer for ChemGPT.
        model: The ChemGPT model.
        device: The device the model is loaded onto.
    """
    try:
        model_name = "ncfrey/ChemGPT-4.7M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading ChemGPT model and tokenizer: {e}")
        st.stop()

# Load models
chemberta_tokenizer, chemberta_model, chemberta_device = load_chemberta_model()
chemgpt_tokenizer, chemgpt_model, chemgpt_device = load_chemgpt_model()

# ===========================
# 7. File Upload and Data Processing
# ===========================

st.header("📂 Upload LogP-LogS CSV File")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data_smiles = pd.read_csv(uploaded_file)
        required_columns = {'Compound ID', 'InChIKey', 'SMILES', 'logS', 'logP', 'MW'}
        if not required_columns.issubset(set(data_smiles.columns)):
            st.error(f"Uploaded CSV must contain the following columns: {required_columns}")
            st.stop()
        
        # Drop rows with missing SMILES or target values
        data_smiles = data_smiles.dropna(subset=['SMILES', 'logP', 'logS'])
        
        st.success("File uploaded and validated successfully!")
        
        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(data_smiles.head())
        
        # Initialize Session State for models
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
        if 'tokenizer' not in st.session_state:
            st.session_state.tokenizer = chemberta_tokenizer
        if 'device' not in st.session_state:
            st.session_state.device = chemberta_device
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.stop()
else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# ===========================
# 8. Molecule Visualization
# ===========================

st.header("🧫 Molecule Visualization")

# Display molecules from the uploaded data
st.subheader("🔹 Input Molecules")
num_molecules_to_display = st.slider("Select number of molecules to display", min_value=1, max_value=20, value=5)

for idx, row in data_smiles.head(num_molecules_to_display).iterrows():
    smiles = row['SMILES']
    img = draw_molecule(smiles)
    if img:
        st.image(img, caption=f"Compound ID: {row['Compound ID']} | InChIKey: {row['InChIKey']}")
    else:
        st.write(f"⚠️ Could not parse molecule: {smiles}")

# ===========================
# 9. Fine-Tune ChemBERTa for Regression
# ===========================

st.header("📈 Fine-Tune ChemBERTa for LogP and LogS Prediction")

st.markdown("""
Fine-tune the ChemBERTa model to predict `LogP` and `LogS` values directly from SMILES strings.
""")

# Training options
train_model = st.checkbox("Fine-Tune ChemBERTa Model on Uploaded Data")

if train_model:
    st.subheader("🔧 Training Configuration")
    
    # Define training parameters
    num_epochs = st.number_input("Number of Training Epochs", min_value=1, max_value=20, value=3, step=1)
    train_batch_size = st.number_input("Training Batch Size", min_value=1, max_value=64, value=16, step=1)
    eval_batch_size = st.number_input("Evaluation Batch Size", min_value=1, max_value=64, value=16, step=1)
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-3, value=2e-5, step=1e-6, format="%.6f")
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1.0, value=0.01, step=0.01, format="%.2f")
    
    # Start fine-tuning
    if st.button("Start Fine-Tuning"):
        with st.spinner("Fine-tuning ChemBERTa... This may take a while."):
            try:
                # Prepare data for fine-tuning
                logger.info("Preparing data for fine-tuning...")
                from transformers import Trainer, TrainingArguments
                
                # Tokenize data
                encodings = chemberta_tokenizer(
                    data_smiles['SMILES'].tolist(),
                    truncation=True,
                    padding=True,
                    max_length=200,
                    return_tensors='pt'
                )
                
                labels = data_smiles[['logP', 'logS']].values.astype(float)
                
                # Create Dataset
                class RegressionDataset(Dataset):
                    def __init__(self, encodings, labels):
                        self.encodings = encodings
                        self.labels = labels

                    def __len__(self):
                        return len(self.labels)

                    def __getitem__(self, idx):
                        item = {key: val[idx] for key, val in self.encodings.items()}
                        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                        return item

                dataset = RegressionDataset(encodings, labels)
                
                # Split into train and eval
                logger.info("Splitting data into training and evaluation sets...")
                train_size = int(0.8 * len(dataset))
                eval_size = len(dataset) - train_size
                train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
                logger.info(f"Training samples: {train_size}, Evaluation samples: {eval_size}")
                
                # Define model
                model = ChemBERTaForMultiOutputRegression(model_name="seyonec/ChemBERTa-zinc-base-v1")
                model.to(chemberta_device)
                
                # Define training arguments
                training_args = TrainingArguments(
                    output_dir='./results',
                    num_train_epochs=int(num_epochs),
                    per_device_train_batch_size=int(train_batch_size),
                    per_device_eval_batch_size=int(eval_batch_size),
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    evaluation_strategy="epoch",
                    logging_dir='./logs',
                    logging_steps=10,
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="mse_logp",
                    greater_is_better=False,
                )
                
                # Define metrics
                def compute_metrics(pred):
                    labels = pred.label_ids
                    preds = pred.predictions
                    mse_logp = mean_squared_error(labels[:,0], preds[:,0])
                    mse_logs = mean_squared_error(labels[:,1], preds[:,1])
                    mae_logp = mean_absolute_error(labels[:,0], preds[:,0])
                    mae_logs = mean_absolute_error(labels[:,1], preds[:,1])
                    return {
                        'mse_logp': mse_logp,
                        'mse_logs': mse_logs,
                        'mae_logp': mae_logp,
                        'mae_logs': mae_logs
                    }
                
                # Initialize Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics,
                    data_collator=default_data_collator
                )
                
                # Train the model
                logger.info("Starting training...")
                trainer.train()
                logger.info("Training completed.")
                
                # Evaluate the model
                eval_results = trainer.evaluate()
                logger.info(f"Evaluation results: {eval_results}")
                
                st.success("Fine-tuning completed successfully!")
                st.write("**Evaluation Metrics:**")
                st.write(f"**LogP - MSE:** {eval_results['eval_mse_logp']:.4f}")
                st.write(f"**LogP - MAE:** {eval_results['eval_mae_logp']:.4f}")
                st.write(f"**LogS - MSE:** {eval_results['eval_mse_logs']:.4f}")
                st.write(f"**LogS - MAE:** {eval_results['eval_mae_logs']:.4f}")
                
                # Store the trained model in session state
                st.session_state.trained_model = model
                
                # Generate predictions for scatter plot
                logger.info("Generating predictions for scatter plot...")
                predictions = trainer.predict(eval_dataset)
                preds = predictions.predictions
                labels = predictions.label_ids
                
                # Store predictions and labels in session state
                st.session_state.predictions = preds
                st.session_state.labels = labels
                
                # Generate scatter plots
                logger.info("Generating scatter plots...")
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Scatter plot for LogP
                sns.scatterplot(x=labels[:,0], y=preds[:,0], ax=axes[0], color='blue', edgecolor='w', s=70)
                min_logp = min(labels[:,0].min(), preds[:,0].min())
                max_logp = max(labels[:,0].max(), preds[:,0].max())
                axes[0].plot([min_logp, max_logp], [min_logp, max_logp], 'r--')
                axes[0].set_xlabel('Actual LogP')
                axes[0].set_ylabel('Predicted LogP')
                axes[0].set_title('LogP: Actual vs Predicted')
                
                # Scatter plot for LogS
                sns.scatterplot(x=labels[:,1], y=preds[:,1], ax=axes[1], color='green', edgecolor='w', s=70)
                min_logs = min(labels[:,1].min(), preds[:,1].min())
                max_logs = max(labels[:,1].max(), preds[:,1].max())
                axes[1].plot([min_logs, max_logs], [min_logs, max_logs], 'r--')
                axes[1].set_xlabel('Actual LogS')
                axes[1].set_ylabel('Predicted LogS')
                axes[1].set_title('LogS: Actual vs Predicted')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"An error occurred during fine-tuning: {e}")

    st.markdown("---")

# ===========================
# 10. Property Prediction Using Fine-Tuned Model
# ===========================

st.header("📊 Predict Molecular Properties with Fine-Tuned ChemBERTa")

st.markdown("""
Use the fine-tuned ChemBERTa model to predict `LogP` and `LogS` values for new SMILES strings.
""")

# Prediction options
predict_properties_btn = st.button("Predict Properties")

if predict_properties_btn:
    # Check if the model is trained
    if st.session_state.trained_model is None:
        st.warning("Please fine-tune the model before making predictions.")
    else:
        # Input SMILES
        new_smiles = st.text_area("Enter SMILES strings (one per line):", height=150)
        smiles_list = [s.strip() for s in new_smiles.splitlines() if s.strip()]
        
        if smiles_list:
            try:
                # Validate SMILES
                valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
                invalid_smiles = [s for s in smiles_list if not is_valid_smiles(s)]
                
                if invalid_smiles:
                    st.warning(f"The following SMILES strings are invalid and will be skipped:\n" + "\n".join(invalid_smiles))
                
                if not valid_smiles:
                    st.error("No valid SMILES strings provided for prediction.")
                else:
                    # Tokenize input
                    encodings = chemberta_tokenizer(
                        valid_smiles,
                        truncation=True,
                        padding=True,
                        max_length=200,
                        return_tensors='pt'
                    )
                    input_ids = encodings['input_ids'].to(st.session_state.device)
                    attention_mask = encodings['attention_mask'].to(st.session_state.device)
                    
                    # Get predictions
                    st.info("Making predictions...")
                    with torch.no_grad():
                        model_output = st.session_state.trained_model(input_ids=input_ids, attention_mask=attention_mask)
                        if isinstance(model_output, tuple):
                            preds = model_output[1].cpu().numpy()  # Predictions
                        else:
                            preds = model_output.cpu().numpy()
                    
                    # Handle single prediction case
                    if preds.ndim == 1:
                        preds = preds.reshape(1, -1)
                    
                    # Prepare results
                    results = pd.DataFrame({
                        "SMILES": valid_smiles,
                        "Predicted LogP": preds[:,0],
                        "Predicted LogS": preds[:,1]
                    })
                    
                    st.subheader("🔍 Prediction Results")
                    st.dataframe(results)
                    
                    # Optionally, download the predictions
                    if st.checkbox("Download Prediction Results as CSV"):
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name='property_predictions.csv',
                            mime='text/csv'
                        )
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error("Please enter at least one SMILES string for prediction.")
else:
    st.info("Click the button above to predict properties for new SMILES strings.")

st.markdown("---")

# ===========================
# 11. Molecule Generation Handling
# ===========================

st.header("🧪 Molecule Generation with ChemGPT")

st.markdown("""
Generate new molecules based on input SMILES strings using ChemGPT. Ensure that the model used supports causal language modeling.
""")

# Function to generate molecules using ChemGPT
def generate_molecules(smiles_list, chemgpt_tokenizer, chemgpt_model, num_generations=5):
    """
    Generate new molecules based on input SMILES strings using ChemGPT.
    """
    generations = {}
    output = []
    
    for mol in smiles_list:
        data = {}
        try:
            # Encode the input SMILES
            input_ids = chemgpt_tokenizer.encode(mol, return_tensors="pt").to(chemgpt_device)
            
            # Generate new SMILES strings using Sampling
            generated_ids = chemgpt_model.generate(
                input_ids,
                max_length=200,  # Adjust as needed
                num_return_sequences=num_generations,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            # Decode the generated SMILES
            generated_smiles = [
                Chem.CanonSmiles(chemgpt_tokenizer.decode(g_id, skip_special_tokens=True).strip())
                for g_id in generated_ids
            ]
            
            # Filter out invalid SMILES
            valid_generated_smiles = [s for s in generated_smiles if Chem.MolFromSmiles(s)]
            
            if not valid_generated_smiles:
                st.warning(f"No valid generated SMILES for input: {mol}")
                continue
            
            # Calculate Tanimoto similarity
            fingerprints_input = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(mol))
            chemgenerations = []
            
            for gen_smiles in valid_generated_smiles:
                fingerprint_gen = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(gen_smiles))
                similarity = DataStructs.TanimotoSimilarity(fingerprints_input, fingerprint_gen)
                chemgenerations.append({
                    "Generated Molecule": gen_smiles,
                    "Tanimoto Similarity": round(similarity, 4)
                })
            
            data['Input Molecule'] = Chem.CanonSmiles(mol)
            data['Generated Molecules'] = chemgenerations
            output.append(data)
        
        except Exception as e:
            st.error(f"Error generating molecules for SMILES {mol}: {e}")
    
    generations['data'] = output
    return generations

# Function to display generated molecules
def display_generations(generations):
    if 'data' in generations and generations['data']:
        for data in generations['data']:
            st.subheader(f"Input Molecule: {data['Input Molecule']}")
            st.write("**Generated Molecules:**")
            gen_df = pd.DataFrame(data['Generated Molecules'])
            st.table(gen_df)
            st.markdown("---")
    else:
        st.write("No generations to display.")

# Molecule generation button
generate_btn = st.button("Generate Molecules with ChemGPT")

if generate_btn:
    # Input SMILES for generation
    generation_smiles = st.text_area("Enter SMILES strings for Molecule Generation (one per line):", height=150)
    smiles_list = [s.strip() for s in generation_smiles.splitlines() if s.strip()]
    
    if smiles_list:
        try:
            # Validate SMILES
            valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
            invalid_smiles = [s for s in smiles_list if not is_valid_smiles(s)]
            
            if invalid_smiles:
                st.warning(f"The following SMILES strings are invalid and will be skipped:\n" + "\n".join(invalid_smiles))
            
            if not valid_smiles:
                st.error("No valid SMILES strings provided for molecule generation.")
            else:
                # Generate molecules using Sampling Approach
                generations = generate_molecules(valid_smiles, chemgpt_tokenizer, chemgpt_model, num_generations=5)
                
                st.success("Molecule generation completed!")
                
                # Display generations
                display_generations(generations)
                
                # Optionally, allow downloading the generations
                if st.checkbox("Download Molecule Generations as JSON"):
                    json_data = json.dumps(generations, indent=4)
                    st.download_button(
                        label="Download Generations",
                        data=json_data,
                        file_name='chemgpt_generations.json',
                        mime='application/json'
                    )
        except Exception as e:
            st.error(f"An error occurred during molecule generation: {e}")
    else:
        st.error("Please enter at least one SMILES string for molecule generation.")
else:
    st.info("Click the button above to generate molecules based on the provided SMILES strings.")

# ===========================
# 12. Footer
# ===========================

st.markdown("---")
st.markdown("© 2024 Smart Reaction Design Tool | Powered by Sambanova, Hugging Face, and RDKit")
