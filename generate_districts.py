import pandas as pd
import pickle
import requests
import time
from typing import Dict, List

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

def main():
    print("Loading existing results...")
    try:
        with open('results.pkl', 'rb') as f:
            results_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return

    results_2025 = results_data['results_2025']
    total_programs = len(results_2025)
    
    print(f"Processing {total_programs} programs...")
    
    # Create a dictionary to cache district lookups
    district_cache = {}
    
    for idx, result in enumerate(results_2025):
        print(f"Processing program {idx + 1} of {total_programs}")
        
        company_name = result.get("Matched Company/Institution", "")
        if not company_name:
            continue
            
        # Check cache first
        if company_name in district_cache:
            district = district_cache[company_name]
        else:
            district = get_company_district(company_name)
            district_cache[company_name] = district
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
        
        # Update the result with district information
        result["District"] = district
    
    # Save updated results
    print("Saving updated results...")
    with open('results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print("District information updated successfully!")

if __name__ == "__main__":
    main() 