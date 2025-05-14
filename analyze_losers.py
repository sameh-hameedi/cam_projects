import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict, Tuple

def load_data():
    """Load required data files."""
    try:
        # Load PDF programs
        with open('pdf_programs.pkl', 'rb') as f:
            pdf_programs = pickle.load(f)
            
        # Load 2025 results
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        # Load embeddings
        with open('embeddings.pkl', 'rb') as f:
            embeddings_data = pickle.load(f)
            
        return pdf_programs, results, embeddings_data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

def find_similar_programs(program_desc: str, embeddings_data: Dict, model: SentenceTransformer, threshold: float = 0.75) -> List[Tuple[str, float, int]]:
    """Find similar programs using pre-computed embeddings."""
    if not program_desc or not embeddings_data:
        return []
        
    try:
        # Get embeddings for 2025 programs
        programs_2025 = embeddings_data['programs_2025']
        embeddings_2025 = embeddings_data['embeddings_2025']
        
        # Encode the current program
        program_embedding = model.encode(program_desc)
        
        # Calculate similarities
        similarities = np.dot(embeddings_2025, program_embedding) / (
            np.linalg.norm(embeddings_2025, axis=1) * np.linalg.norm(program_embedding)
        )
        
        similar_programs = [(prog, sim, idx) for idx, (prog, sim) in enumerate(zip(programs_2025, similarities)) if sim > threshold]
        return sorted(similar_programs, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Error finding similar programs: {str(e)}")
        return []

def analyze_programs(pdf_programs: List[Dict], embeddings_data: Dict, model: SentenceTransformer) -> List[Dict]:
    """Analyze programs and include similarity scores for all programs."""
    analyzed_programs = []
    
    for program_data in pdf_programs:
        program = program_data['program']
        requested_amount = program_data['requested_amount']
        
        # Find similar programs
        similar_programs = find_similar_programs(program, embeddings_data, model)
        
        # Get the highest similarity score if any similar programs found
        similarity = similar_programs[0][1] if similar_programs else 0.0
        
        analyzed_programs.append({
            'program': program,
            'requested_amount': requested_amount,
            'similarity': similarity
        })
    
    return analyzed_programs

def main():
    # Load data
    pdf_programs, results, embeddings_data = load_data()
    if not all([pdf_programs, results, embeddings_data]):
        print("Failed to load required data")
        return
    
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Analyze House programs
    house_programs = analyze_programs(pdf_programs['house_programs'], embeddings_data, model)
    
    # Analyze Senate programs
    senate_programs = analyze_programs(pdf_programs['senate_programs'], embeddings_data, model)
    
    # Save analysis results
    analysis = {
        'house_programs': house_programs,
        'senate_programs': senate_programs
    }
    
    with open('losers_analysis.pkl', 'wb') as f:
        pickle.dump(analysis, f)
    
    print("Analysis complete!")
    print(f"Analyzed {len(house_programs)} programs in House PDF")
    print(f"Analyzed {len(senate_programs)} programs in Senate PDF")

if __name__ == "__main__":
    main() 