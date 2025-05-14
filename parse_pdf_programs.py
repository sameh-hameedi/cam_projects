import pdf2image
import pytesseract
import json
import requests
import time
import os
import pickle
from typing import List, Dict
import re
from PIL import Image

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from PDF file page by page using OCR."""
    pages = []
    
    # Convert PDF to images
    print(f"Converting {pdf_path} to images...")
    images = pdf2image.convert_from_path(pdf_path)
    
    for i, image in enumerate(images):
        print(f"Processing page {i+1}/{len(images)} with OCR...")
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        if text.strip():
            pages.append(text)
    
    return pages

def extract_programs(text: str) -> List[Dict[str, str]]:
    """Extract program names and their requested amounts from text."""
    programs = []
    
    # Split text into lines
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line ends with a large number (3 or more digits)
        if re.search(r'\d{3,}(?:,\d{3})*(?:\.\d+)?$', line):
            # Remove program code (4 alphanumeric characters at start)
            program = re.sub(r'^[A-Z0-9]{4}\s+', '', line)
            
            # Extract the first number (requested amount)
            amount_match = re.search(r'\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', program)
            requested_amount = amount_match.group(1) if amount_match else ""
            
            # Remove all numbers and text after them
            program_name = re.sub(r'\s+\d{1,3}(?:,\d{3})*(?:\.\d+)?.*$', '', program)
            
            # Clean up extra spaces and remove trailing periods
            program_name = ' '.join(program_name.split()).rstrip('.')
            
            if program_name and len(program_name) > 3:  # Basic validation
                programs.append({
                    'program': program_name,
                    'requested_amount': requested_amount
                })
    
    return programs

def main():
    # Process House PDF
    print("Processing House PDF...")
    house_pages = extract_text_from_pdf("house_pdf_2025.pdf")
    house_programs = []
    
    for i, page in enumerate(house_pages):
        print(f"Processing House page {i+1}/{len(house_pages)}")
        programs = extract_programs(page)
        house_programs.extend(programs)
    
    # Process Senate PDF
    print("Processing Senate PDF...")
    senate_pages = extract_text_from_pdf("senate_pdf_2025.pdf")
    senate_programs = []
    
    for i, page in enumerate(senate_pages):
        print(f"Processing Senate page {i+1}/{len(senate_pages)}")
        programs = extract_programs(page)
        senate_programs.extend(programs)
    
    # Remove duplicates while preserving order
    house_programs = list({p['program']: p for p in house_programs}.values())
    senate_programs = list({p['program']: p for p in senate_programs}.values())
    
    # Save results
    results = {
        'house_programs': house_programs,
        'senate_programs': senate_programs
    }
    
    with open('pdf_programs.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("PDF processing complete!")
    print(f"Found {len(house_programs)} programs in House PDF")
    print(f"Found {len(senate_programs)} programs in Senate PDF")
    
    # Print some examples
    print("\nExample programs from House PDF:")
    for prog in house_programs[:5]:
        print(f"- {prog['program']} (Requested: {prog['requested_amount']})")
    
    print("\nExample programs from Senate PDF:")
    for prog in senate_programs[:5]:
        print(f"- {prog['program']} (Requested: {prog['requested_amount']})")

if __name__ == "__main__":
    main() 