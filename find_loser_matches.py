import pickle
import requests
import time
from typing import List, Dict
from datetime import datetime
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def clean_amount(amount_str: str) -> float:
    """Clean and convert amount string to float."""
    if not amount_str:
        return 0.0
    # Remove $ and commas, then convert to float
    return float(re.sub(r'[$,]', '', amount_str))

def generate_keywords(program_desc: str) -> List[str]:
    """Generate keywords from program description."""
    if not program_desc:
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

def search_recipient_id(company_name: str) -> str:
    """Search for recipient ID using USAspending API."""
    base_url = "https://api.usaspending.gov/api/v2/recipient/"
    
    payload = {
        "order": "desc",
        "sort": "amount",
        "page": 1,
        "limit": 1,
        "keyword": company_name,
        "award_type": "all"
    }
    
    try:
        response = session.post(base_url, json=payload)
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
    if not recipient_id:
        return {}
        
    base_url = f"https://api.usaspending.gov/api/v2/recipient/{recipient_id}/"
    
    try:
        response = session.get(base_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting recipient details: {str(e)}")
        return {}

def get_company_district(company_name: str) -> str:
    """Get district information for a company."""
    recipient_id = search_recipient_id(company_name)
    if not recipient_id:
        return ""
        
    details = get_recipient_details(recipient_id)
    if not details:
        return ""
        
    # Extract state and congressional district using the same format as winners
    location = details.get('location', {})
    if location:
        state = location.get('state_code', '')
        district = location.get('congressional_code', '')
        if state and district:
            return f"{state}-{district}"
    
    return ""

def query_usaspending(keywords: List[str], requested_amount: float) -> List[Dict]:
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
    lower_bound = requested_amount * 0.75  # 25% below
    upper_bound = requested_amount * 1.25  # 25% above
    
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
        response = session.post(base_url, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        # Add district information to each result
        for result in results:
            company_name = result.get("Recipient Name", "")
            if company_name:
                result["District"] = get_company_district(company_name)
                time.sleep(2)  # Increased delay between recipient API calls
        
        return results
    except Exception as e:
        print(f"Error querying USAspending.gov: {str(e)}")
        return []

def process_programs(programs: List[Dict]) -> List[Dict]:
    """Process programs to find potential matches."""
    processed_programs = []
    
    for program in programs:
        print(f"Processing program: {program['program']}")
        
        # Generate keywords
        keywords = generate_keywords(program['program'])
        
        # Query USAspending
        matches = query_usaspending(keywords, clean_amount(program['requested_amount']))
        
        # Add matches to program data
        program_data = program.copy()
        program_data['potential_matches'] = matches
        
        processed_programs.append(program_data)
        
        # Add delay to avoid rate limits
        time.sleep(2)  # Increased delay between program processing
    
    return processed_programs

def main():
    # Load losers analysis
    try:
        with open('losers_analysis.pkl', 'rb') as f:
            analysis = pickle.load(f)
    except FileNotFoundError:
        print("Error: losers_analysis.pkl not found")
        return
    
    # Process House programs
    print("Processing House programs...")
    house_programs = process_programs(analysis['house_programs'])
    
    # Process Senate programs
    print("Processing Senate programs...")
    senate_programs = process_programs(analysis['senate_programs'])
    
    # Save results
    results = {
        'house_programs': house_programs,
        'senate_programs': senate_programs
    }
    
    with open('loser_matches.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Analysis complete!")
    print(f"Processed {len(house_programs)} House programs")
    print(f"Processed {len(senate_programs)} Senate programs")

if __name__ == "__main__":
    main() 