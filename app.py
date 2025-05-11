import streamlit as st
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
from datetime import datetime, timedelta

# Initialize the sentence transformer model
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    df_2025 = pd.read_csv("pentagon_budget_increases_2025.csv")
    df_2024 = pd.read_csv("pentagon_budget_increases_2024.csv")
    # Take only first 10 rows of 2025 data for testing
    df_2025 = df_2025.head(10)
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

def find_similar_programs(program_desc: str, model, all_programs: List[str], threshold: float = 0.75) -> List[Tuple[str, float, int]]:
    """Find similar programs using embeddings."""
    if not program_desc or not all_programs:
        return []
        
    # Filter out None values and empty strings
    valid_programs = [p for p in all_programs if p and not pd.isna(p)]
    if not valid_programs:
        return []
        
    try:
        program_embedding = model.encode(program_desc)
        all_embeddings = model.encode(valid_programs)
        
        similarities = np.dot(all_embeddings, program_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(program_embedding)
        )
        
        similar_programs = [(prog, sim, idx) for idx, (prog, sim) in enumerate(zip(valid_programs, similarities)) if sim > threshold]
        return sorted(similar_programs, key=lambda x: x[1], reverse=True)
    except Exception as e:
        st.error(f"Error finding similar programs: {str(e)}")
        return []

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    filtered_df = df.copy()
    for column, value in filters.items():
        if value and value != "All":
            filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(value, case=False)]
    return filtered_df

def main():
    st.title("Defense Contract Analysis Tool")
    st.write("Testing with first 10 rows of 2025 data")
    
    # Add a progress bar
    progress_bar = st.progress(0)
    
    # Load data
    df_2025, df_2024 = load_data()
    
    # Load model
    model = load_model()
    
    # Process amounts for both years
    df_2025["House Amount"] = df_2025["House Amount (in thousands)"].apply(clean_amount)
    df_2025["Senate Amount"] = df_2025["Senate Amount (in thousands)"].apply(clean_amount)
    df_2024["House Amount"] = df_2024["House Amount (in thousands)"].apply(clean_amount)
    df_2024["Senate Amount"] = df_2024["Senate Amount (in thousands)"].apply(clean_amount)
    
    # Generate embeddings for all programs
    all_programs_2024 = df_2024["Program Increase"].dropna().tolist()
    
    # Create results container
    results_2025 = []
    
    # Add a status message
    status_text = st.empty()
    
    # Process each 2025 program
    total_programs = len(df_2025)
    for idx, (_, row) in enumerate(df_2025.iterrows()):
        # Update progress
        progress = (idx + 1) / total_programs
        progress_bar.progress(progress)
        status_text.text(f"Processing program {idx + 1} of {total_programs}")
        
        program_desc = row["Program Increase"]
        if pd.isna(program_desc):
            continue
            
        # Find similar programs from 2024
        similar_programs = find_similar_programs(program_desc, model, all_programs_2024)
        
        # Generate keywords and query USAspending
        keywords = generate_keywords(program_desc)
        awards = query_usaspending(keywords, row["House Amount"], row["Senate Amount"])
        
        # Find best matching award
        best_match = None
        award_amount = 0
        if awards:
            total_budget = row["House Amount"] + row["Senate Amount"]
            
            # First try to find a match based on description
            award_descriptions = [award.get("Award Description", "") for award in awards if award.get("Award Description")]
            if award_descriptions:
                award_similarities = find_similar_programs(program_desc, model, award_descriptions)
                if award_similarities:
                    best_match_idx = award_descriptions.index(award_similarities[0][0])
                    best_match = awards[best_match_idx]
                    award_amount = best_match.get("Award Amount", 0)
            
            # If no description match found, find the award with amount closest to total budget
            if not best_match and awards:
                # Calculate difference between each award amount and total budget
                amount_differences = [
                    (abs(award.get("Award Amount", 0) - total_budget), award)
                    for award in awards
                ]
                # Sort by difference and take the closest match
                amount_differences.sort(key=lambda x: x[0])
                best_match = amount_differences[0][1]
                award_amount = best_match.get("Award Amount", 0)
        
        # Calculate percentage deviation
        total_budget = row["House Amount"] + row["Senate Amount"]
        if total_budget > 0:
            deviation = ((award_amount - total_budget) / total_budget) * 100
        else:
            deviation = 0
        
        # Add to results
        result = {
            "Program Description": program_desc,
            "Matched Company/Institution": best_match["Recipient Name"] if best_match else "",
            "Matched Place of Performance": best_match["Place of Performance"] if best_match else "",
            "Award Amount": f"${award_amount:,.2f}",
            "Matched Award Deviation": f"{deviation:+.1f}%",  # + for positive, - for negative
            "House Amount": f"${row['House Amount']:,.2f}",
            "Senate Amount": f"${row['Senate Amount']:,.2f}",
            "Similar 2024 Programs": [f"Row {idx+1}: {prog}" for prog, _, idx in similar_programs[:3]]
        }
        results_2025.append(result)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create tabs for 2025 and 2024 data
    tab1, tab2 = st.tabs(["2025 Programs", "2024 Programs"])
    
    with tab1:
        st.header("2025 Programs Analysis")
        if results_2025:
            df_results_2025 = pd.DataFrame(results_2025)
            
            # Add filters
            st.subheader("Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                program_filter = st.text_input("Filter by Program Description", "")
                company_filter = st.text_input("Filter by Company/Institution", "")
                location_filter = st.text_input("Filter by Place of Performance", "")
            
            with col2:
                house_amount_filter = st.text_input("Filter by House Amount", "")
                senate_amount_filter = st.text_input("Filter by Senate Amount", "")
                similar_programs_filter = st.text_input("Filter by Similar 2024 Programs", "")
            
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
                height=400,
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
                    "Matched Place of Performance": st.column_config.TextColumn(
                        "Matched Place of Performance",
                        width="medium",
                        help="Location where the work will be performed"
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
                }
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
        
        # Add filters for 2024 data
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            program_filter_2024 = st.text_input("Filter by Program Description (2024)", "")
        
        with col2:
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
            height=400,
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