import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import re
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict, Tuple

def clean_amount(amount_str: str) -> float:
    """Clean and convert amount string to float."""
    if pd.isna(amount_str):
        return 0.0
    # Remove $ and commas, then convert to float
    return float(re.sub(r'[$,]', '', amount_str))

def search_recipient_id(company_name: str) -> str:
    """Search for a recipient ID using the search endpoint."""
    search_url = "https://api.usaspending.gov/api/v2/recipient/"
    
    payload = {
        "order": "desc",
        "sort": "amount",
        "page": 1,
        "limit": 1,
        "keyword": company_name,
        "award_type": "all"
    }
    
    try:
        response = requests.post(search_url, json=payload)
        response.raise_for_status()
        results = response.json()
        
        if results.get("results"):
            return results["results"][0]["id"]
        return None
    except Exception as e:
        print(f"Error searching for recipient ID: {str(e)}")
        return None

def get_recipient_details(recipient_id: str) -> dict:
    """Get detailed information about a recipient using their ID."""
    detail_url = f"https://api.usaspending.gov/api/v2/recipient/{recipient_id}/"
    
    try:
        response = requests.get(detail_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting recipient details: {str(e)}")
        return None

def get_company_district(company_name: str) -> str:
    """Get company's district information by combining state and congressional district."""
    if pd.isna(company_name) or not company_name:
        return ""
        
    # First get the recipient ID
    recipient_id = search_recipient_id(company_name)
    if not recipient_id:
        return ""
    
    # Then get the detailed information
    details = get_recipient_details(recipient_id)
    if not details:
        return ""
    
    location = details.get('location', {})
    if location:
        state = location.get('state_code', '')
        district = location.get('congressional_code', '')
        if state and district:
            return f"{state}-{district}"
    
    return ""

def generate_keywords(program_desc: str) -> List[str]:
    """Generate keywords from program description."""
    if not program_desc or pd.isna(program_desc):
        return []
        
    # Split into words and create combinations
    words = program_desc.split()
    keywords = [program_desc]  # Add full description
    
    # Add individual words if they're not too short
    keywords.extend([w for w in words if len(w) > 3])
    
    # Add 2-word combinations
    for i in range(len(words)-1):
        keywords.append(f"{words[i]} {words[i+1]}")
    
    return list(set(keywords))  # Remove duplicates

def query_usaspending(keywords: List[str], house_amount: float, senate_amount: float) -> List[Dict]:
    """Query USAspending.gov API for matching awards."""
    if not keywords:
        return []
        
    base_url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    
    # Get current fiscal year
    current_date = datetime.now()
    if current_date.month >= 10:
        start_date = f"{current_date.year-5}-10-01"
        end_date = f"{current_date.year + 10}-09-30"
    else:
        start_date = f"{current_date.year - 5}-10-01"
        end_date = f"{current_date.year+10}-09-30"
    
    # Calculate amount range for matching
    total_amount = house_amount + senate_amount
    lower_bound = total_amount * 0.75  # 25% below
    upper_bound = total_amount * 1.25  # 25% above
    
    # Create a more specific search query
    search_text = " ".join(keywords)
    
    payload = {
        "subawards": False,
        "limit": 10,
        "page": 1,
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],  # Added D for IDVs
            "time_period": [{"start_date": start_date, "end_date": end_date}],
            "keywords": keywords,
            "naics_codes": ["541715", "541330", "541712", "541711"],  # Common defense-related NAICS codes
            "award_amounts": [
                {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                }
            ]
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Recipient Address",
            "Award Description",
            "Award Amount",
            "Place of Performance",
            "NAICS Code",
            "NAICS Description",
            "Period of Performance Start Date",
            "Period of Performance End Date",
            "Award Type",
            "Awarding Agency",
            "Awarding Sub Agency"
        ]
    }
    
    try:
        response = requests.post(base_url, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
        return results
    except Exception as e:
        print(f"Error querying USAspending.gov: {str(e)}")
        return []

def find_similar_programs(program_desc: str, embeddings_data: Dict, threshold: float = 0.75) -> List[Tuple[str, float, int]]:
    """Find similar programs using pre-computed embeddings."""
    if not program_desc or not embeddings_data:
        return []
        
    try:
        # Get embeddings for 2024 programs
        programs_2024 = embeddings_data['programs_2024']
        embeddings_2024 = embeddings_data['embeddings_2024']
        
        # Find the index of the current program in 2025 data
        programs_2025 = embeddings_data['programs_2025']
        embeddings_2025 = embeddings_data['embeddings_2025']
        
        if program_desc in programs_2025:
            idx = programs_2025.index(program_desc)
            program_embedding = embeddings_2025[idx]
            
            # Calculate similarities with 2024 programs
            similarities = np.dot(embeddings_2024, program_embedding) / (
                np.linalg.norm(embeddings_2024, axis=1) * np.linalg.norm(program_embedding)
            )
            
            similar_programs = [(prog, sim, idx) for idx, (prog, sim) in enumerate(zip(programs_2024, similarities)) if sim > threshold]
            return sorted(similar_programs, key=lambda x: x[1], reverse=True)
        return []
    except Exception as e:
        print(f"Error finding similar programs: {str(e)}")
        return []

def main():
    print("Loading data...")
    # Load data
    df_2025 = pd.read_csv("pentagon_budget_increases_2025.csv")
    df_2024 = pd.read_csv("pentagon_budget_increases_2024.csv")
    
    # Process amounts
    df_2025["House Amount"] = df_2025["House Amount"].apply(clean_amount)
    df_2025["Senate Amount"] = df_2025["Senate Amount"].apply(clean_amount)
    df_2024["House Amount"] = df_2024["House Amount"].apply(clean_amount)
    df_2024["Senate Amount"] = df_2024["Senate Amount"].apply(clean_amount)
    
    print("Loading model...")
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for 2024 programs
    programs_2024 = df_2024["Program Increase"].tolist()
    embeddings_2024 = model.encode(programs_2024)
    
    # Generate embeddings for 2025 programs
    programs_2025 = df_2025["Program Increase"].tolist()
    embeddings_2025 = model.encode(programs_2025)
    
    # Save embeddings
    embeddings_data = {
        'programs_2024': programs_2024,
        'embeddings_2024': embeddings_2024,
        'programs_2025': programs_2025,
        'embeddings_2025': embeddings_2025
    }
    
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print("Generating results for 2025 programs...")
    # Process each 2025 program
    results_2025 = []
    total_programs = len(df_2025)
    
    for idx, row in df_2025.iterrows():
        print(f"Processing program {idx + 1} of {total_programs}")
        
        # Find similar programs
        similar_programs = find_similar_programs(row["Program Increase"], embeddings_data)
        similar_programs_str = [f"{prog} (Row {idx+1})" for prog, _, idx in similar_programs]
        
        # Generate keywords and query USAspending
        keywords = generate_keywords(row["Program Increase"])
        usaspending_results = query_usaspending(keywords, row["House Amount"], row["Senate Amount"])
        
        # Get district information for the matched company
        district = ""
        if usaspending_results:
            company_name = usaspending_results[0].get("Recipient Name", "")
            district = get_company_district(company_name)
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
        
        # Create result entry
        result = {
            "Program Description": row["Program Increase"],
            "House Amount": row["House Amount"],
            "Senate Amount": row["Senate Amount"],
            "Similar 2024 Programs": similar_programs_str,
            "Matched Company/Institution": usaspending_results[0].get("Recipient Name", "") if usaspending_results else "",
            "Matched Place of Performance": usaspending_results[0].get("Place of Performance", "") if usaspending_results else "",
            "Award Amount": usaspending_results[0].get("Award Amount", 0) if usaspending_results else 0,
            "Matched Award Deviation": f"{((usaspending_results[0].get('Award Amount', 0) - (row['House Amount'] + row['Senate Amount'])) / (row['House Amount'] + row['Senate Amount']) * 100):.2f}%" if usaspending_results else "",
            "District": district
        }
        results_2025.append(result)
    
    # Save all results
    results_data = {
        'results_2025': results_2025,
        'df_2024': df_2024,
        'df_2025': df_2025
    }
    
    with open('results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print("All results generated and saved successfully!")

if __name__ == "__main__":
    main() 