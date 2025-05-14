import requests
import pickle
from bs4 import BeautifulSoup
import re

def get_district_party(state: str, district: str) -> str:
    """Get party affiliation for a congressional district from house.gov representatives table."""
    if not state or not district:
        return "Unknown"
        
    # Format state and district
    state = state.upper()
    district = str(district).zfill(2)  # Pad with leading zeros
    
    # Get the representatives page
    url = "https://www.house.gov/representatives"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the state section
        state_header = soup.find(string=re.compile(state, re.IGNORECASE))
        if not state_header:
            return "Unknown"
            
        # Find the table after the state header
        table = state_header.find_next('table')
        if not table:
            return "Unknown"
            
        # Find the row for this district
        district_num = int(district)  # Convert to int to match table format
        for row in table.find_all('tr')[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 3:
                # Extract district number from first column
                dist_text = cols[0].text.strip()
                if dist_text.endswith('st') or dist_text.endswith('nd') or dist_text.endswith('rd') or dist_text.endswith('th'):
                    dist_num = int(dist_text[:-2])  # Remove suffix and convert to int
                    if dist_num == district_num:
                        # Get party from third column
                        party = cols[2].text.strip()
                        return party
                        
        return "Unknown"
    except Exception as e:
        print(f"Error getting party for {state}-{district}: {str(e)}")
        return "Unknown"

def main():
    # Load existing results
    try:
        with open('results.pkl', 'rb') as f:
            results_data = pickle.load(f)
    except FileNotFoundError:
        print("Error: results.pkl not found")
        return
    
    # Create a cache for district parties
    district_parties = {}
    
    # Process each result
    for result in results_data['results_2025']:
        district = result.get("District", "")
        if not district or district in district_parties:
            continue
            
        # Parse state and district
        try:
            state, dist = district.split("-")
            party = get_district_party(state, dist)
            district_parties[district] = party
            print(f"Found party {party} for {state}-{dist}")  # Debug print
        except:
            continue
    
    # Save district parties
    with open('district_parties.pkl', 'wb') as f:
        pickle.dump(district_parties, f)
    
    print("District party information saved!")

if __name__ == "__main__":
    main()
