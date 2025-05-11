import requests
import json

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

def get_company_district(company_name: str) -> tuple:
    """Get company's district information by combining state and congressional district."""
    # First get the recipient ID
    recipient_id = search_recipient_id(company_name)
    if not recipient_id:
        print(f"No recipient ID found for {company_name}")
        return None, None
    
    # Then get the detailed information
    details = get_recipient_details(recipient_id)
    if not details:
        print(f"No details found for recipient ID {recipient_id}")
        return None, None
    
    location = details.get('location', {})
    if location:
        state = location.get('state_code', '')
        district = location.get('congressional_code', '')
        return state, district
    
    return None, None

def main():
    # Test with a company from your data
    test_company = "ROWAN UNIVERSITY"
    
    print(f"Searching for {test_company}...")
    state, district = get_company_district(test_company)
    
    if state and district:
        print(f"\nDistrict Information:")
        print(f"State: {state}")
        print(f"Congressional District: {district}")
        print(f"Combined District: {state}-{district}")
        
        # Get full details for additional information
        recipient_id = search_recipient_id(test_company)
        details = get_recipient_details(recipient_id)
        
        if details:
            print("\nAdditional Details:")
            print(f"Name: {details.get('name')}")
            print(f"UEI: {details.get('uei')}")
            print(f"DUNS: {details.get('duns')}")
            
            location = details.get('location', {})
            if location:
                print("\nFull Address:")
                print(f"{location.get('address_line1', '')}")
                if location.get('address_line2'):
                    print(location.get('address_line2'))
                print(f"{location.get('city_name', '')}, {state} {location.get('zip', '')}")
    else:
        print("No district information found")

if __name__ == "__main__":
    main() 