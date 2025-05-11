import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import re

def clean_amount(amount_str: str) -> float:
    """Clean and convert amount string to float."""
    if pd.isna(amount_str):
        return 0.0
    # Remove $ and commas, then convert to float
    return float(re.sub(r'[$,]', '', amount_str))

def main():
    # Load data
    df_2025 = pd.read_csv("pentagon_budget_increases_2025.csv")
    df_2024 = pd.read_csv("pentagon_budget_increases_2024.csv")
    
    # Take only first 10 rows of 2025 data for testing
    df_2025 = df_2025
    
    # Process amounts
    df_2025["House Amount"] = df_2025["House Amount (in thousands)"].apply(clean_amount)
    df_2025["Senate Amount"] = df_2025["Senate Amount (in thousands)"].apply(clean_amount)
    df_2024["House Amount"] = df_2024["House Amount (in thousands)"].apply(clean_amount)
    df_2024["Senate Amount"] = df_2024["Senate Amount (in thousands)"].apply(clean_amount)
    
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for 2024 programs
    valid_programs_2024 = df_2024["Program Increase"].dropna().tolist()
    embeddings_2024 = model.encode(valid_programs_2024)
    
    # Generate embeddings for 2025 programs
    valid_programs_2025 = df_2025["Program Increase"].dropna().tolist()
    embeddings_2025 = model.encode(valid_programs_2025)
    
    # Save embeddings and program descriptions
    embeddings_data = {
        'programs_2024': valid_programs_2024,
        'embeddings_2024': embeddings_2024,
        'programs_2025': valid_programs_2025,
        'embeddings_2025': embeddings_2025
    }
    
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print("Embeddings generated and saved successfully!")

if __name__ == "__main__":
    main() 