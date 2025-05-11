# Defense Contract Analysis Tool

This Streamlit application analyzes defense contracts from 2024 and 2025, finding similar programs and matching them with companies/institutions that were awarded contracts through USAspending.gov.

## Features

- Compares 2025 defense programs with 2024 programs using semantic similarity
- Generates keywords from program descriptions to search USAspending.gov
- Matches programs with companies/institutions based on award descriptions
- Displays results in an interactive table with program details and matching companies

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## Input Files

The application requires two CSV files:
- `pentagon_budget_increases_2025.csv`
- `pentagon_budget_increases_2024.csv`

These files should contain the following columns:
- House Amount (in thousands)
- Senate Amount (in thousands)
- Program Increase
- Other columns as specified in the data

## Output

The application displays a table with the following columns:
- Program Description
- Company/Institution
- Place of Performance
- House Amount
- Senate Amount
- Similar 2024 Programs

## Notes

- The application uses the SentenceTransformer model 'all-MiniLM-L6-v2' for generating embeddings
- Similarity threshold is set to 75% for matching programs
- USAspending.gov API queries are limited to the current fiscal year
- The application caches the model to improve performance 