import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
from datetime import datetime, timedelta

# Set page config for wide layout
st.set_page_config(
    page_title="Defense Contract Analysis Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the sentence transformer model
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_embeddings():
    """Load pre-computed embeddings."""
    try:
        with open('embeddings.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

def load_data():
    # Use GitHub raw URLs for the data files
    base_url = "https://raw.githubusercontent.com/sameh-hameedi/cam_projects/main/"
    df_2025 = pd.read_csv(base_url + "pentagon_budget_increases_2025.csv")
    df_2024 = pd.read_csv(base_url + "pentagon_budget_increases_2024.csv")
    # Take only first 10 rows of 2025 data for testing
    df_2025 = df_2025
    df_2024 = df_2024
    return df_2025, df_2024

def clean_amount(amount_str: str) -> float:
    """Clean and convert amount string to float."""
    if pd.isna(amount_str):
        return 0.0
    # Remove $ and commas, then convert to float
    return float(re.sub(r'[$,]', '', amount_str))

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
        st.error(f"Error querying USAspending.gov: {str(e)}")
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
        st.error(f"Error finding similar programs: {str(e)}")
        return []

def load_results():
    """Load pre-computed results."""
    try:
        with open('results.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    filtered_df = df.copy()
    for column, value in filters.items():
        if value and value != "All":
            filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(value, case=False)]
    return filtered_df

def main():
    st.title("Defense Contract Analysis Tool")
    
    # Load pre-computed results
    results_data = load_results()
    if results_data is None:
        st.error("Failed to load results. Please ensure results.pkl is present.")
        return
    
    results_2025 = results_data['results_2025']
    df_2024 = results_data['df_2024']
    df_2025 = results_data['df_2025']
    
    # Create tabs for 2025 and 2024 data
    tab1, tab2 = st.tabs(["2025 Programs", "2024 Programs"])
    
    with tab1:
        st.header("2025 Programs Analysis")
        if results_2025:
            df_results_2025 = pd.DataFrame(results_2025)
            
            # Create two columns for filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Program Filters")
                program_filter = st.text_input("Filter by Program Description", "")
                company_filter = st.text_input("Filter by Company/Institution", "")
            
            with col2:
                st.subheader("Location Filters")
                location_filter = st.text_input("Filter by Place of Performance", "")
                similar_programs_filter = st.text_input("Filter by Similar 2024 Programs", "")
            
            with col3:
                st.subheader("Amount Filters")
                house_amount_filter = st.text_input("Filter by House Amount", "")
                senate_amount_filter = st.text_input("Filter by Senate Amount", "")
            
            # Apply filters
            filters = {
                "Program Description": program_filter,
                "Matched Company/Institution": company_filter,
                "Matched Place of Performance": location_filter,
                "House Amount": house_amount_filter,
                "Senate Amount": senate_amount_filter,
                "Similar 2024 Programs": similar_programs_filter
            }
            
            filtered_df_2025 = apply_filters(df_results_2025, filters)
            
            # Display the table with improved formatting
            st.dataframe(
                filtered_df_2025,
                use_container_width=True,
                height=600,
                column_config={
                    "Program Description": st.column_config.TextColumn(
                        "Program Description",
                        width="large",
                        help="Description of the program"
                    ),
                    "Matched Company/Institution": st.column_config.TextColumn(
                        "Matched Company/Institution",
                        width="medium",
                        help="Name of the company or institution"
                    ),
                    "District": st.column_config.TextColumn(
                        "District",
                        width="small",
                        help="Congressional district of the company (State-District)"
                    ),
                    "House Amount": st.column_config.TextColumn(
                        "House Amount",
                        width="small",
                        help="Amount in House bill"
                    ),
                    "Senate Amount": st.column_config.TextColumn(
                        "Senate Amount",
                        width="small",
                        help="Amount in Senate bill"
                    ),
                    "Award Amount": st.column_config.TextColumn(
                        "Award Amount",
                        width="small",
                        help="Amount of the matching award"
                    ),
                    "Matched Award Deviation": st.column_config.TextColumn(
                        "Matched Award Deviation",
                        width="small",
                        help="Percentage difference between Award Amount and total budget"
                    ),
                    "Similar 2024 Programs": st.column_config.ListColumn(
                        "Similar 2024 Programs",
                        width="large",
                        help="Similar programs from 2024 with row numbers"
                    )
                },
                column_order=[
                    "Program Description",
                    "Matched Company/Institution",
                    "District",
                    "House Amount",
                    "Senate Amount",
                    "Award Amount",
                    "Matched Award Deviation",
                    "Similar 2024 Programs"
                ]
            )
            
            # Add download button for 2025 data
            csv_2025 = filtered_df_2025.to_csv(index=False)
            st.download_button(
                label="Download 2025 results as CSV",
                data=csv_2025,
                file_name="defense_contract_analysis_2025.csv",
                mime="text/csv"
            )
        else:
            st.write("No results found for 2025.")
    
    with tab2:
        st.header("2024 Programs")
        # Create 2024 table with selected columns
        df_2024_display = df_2024[["Program Increase", "House Amount", "Senate Amount"]].copy()
        df_2024_display["House Amount"] = df_2024_display["House Amount"].apply(lambda x: f"${x:,.2f}")
        df_2024_display["Senate Amount"] = df_2024_display["Senate Amount"].apply(lambda x: f"${x:,.2f}")
        df_2024_display.index = df_2024_display.index + 1  # Make index 1-based
        df_2024_display.index.name = "Row"
        
        # Create two columns for filters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Program Filters")
            program_filter_2024 = st.text_input("Filter by Program Description (2024)", "")
        
        with col2:
            st.subheader("Amount Filters")
            house_amount_filter_2024 = st.text_input("Filter by House Amount (2024)", "")
            senate_amount_filter_2024 = st.text_input("Filter by Senate Amount (2024)", "")
        
        # Apply filters
        filters_2024 = {
            "Program Increase": program_filter_2024,
            "House Amount": house_amount_filter_2024,
            "Senate Amount": senate_amount_filter_2024
        }
        
        filtered_df_2024 = apply_filters(df_2024_display, filters_2024)
        
        # Display the table with improved formatting
        st.dataframe(
            filtered_df_2024,
            use_container_width=True,
            height=600,
            column_config={
                "Program Increase": st.column_config.TextColumn(
                    "Program Description",
                    width="large",
                    help="Description of the program"
                ),
                "House Amount": st.column_config.TextColumn(
                    "House Amount",
                    width="small",
                    help="Amount in House bill"
                ),
                "Senate Amount": st.column_config.TextColumn(
                    "Senate Amount",
                    width="small",
                    help="Amount in Senate bill"
                )
            }
        )
        
        # Add download button for 2024 data
        csv_2024 = filtered_df_2024.to_csv()
        st.download_button(
            label="Download 2024 data as CSV",
            data=csv_2024,
            file_name="defense_contract_analysis_2024.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 