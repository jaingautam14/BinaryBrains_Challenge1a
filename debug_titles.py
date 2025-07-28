#!/usr/bin/env python3
"""
Debug script to test title extraction from PDFs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_pdfs import PDFProcessor
from pathlib import Path

def test_title_extraction():
    """Test title extraction with specific PDFs"""
    processor = PDFProcessor()
    
    test_files = [
        ("sample_dataset/pdfs/file02.pdf", "Overview Foundation Level Extensions"),
        ("sample_dataset/pdfs/file03.pdf", "RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library")
    ]
    
    print("Title Extraction Test Results:")
    print("=" * 60)
    
    for pdf_path, expected_title in test_files:
        if not Path(pdf_path).exists():
            print(f"❌ File not found: {pdf_path}")
            continue
            
        print(f"\n📄 Testing: {pdf_path}")
        print(f"Expected: {expected_title}")
        
        # Extract text
        pages_text = processor.extract_text_hybrid(Path(pdf_path))
        
        if pages_text:
            # Extract title
            extracted_title = processor.extract_title(pages_text)
            print(f"Extracted: {extracted_title}")
            
            # Check if match
            if extracted_title.strip() == expected_title.strip():
                print("✅ MATCH!")
            else:
                print("❌ NO MATCH")
                
            print(f"Total pages: {len(pages_text)}")
            
            # Show first few lines for debugging
            first_page_text = pages_text[0]['text']
            lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
            print("\nFirst 10 lines:")
            for i, line in enumerate(lines[:10]):
                print(f"  {i+1}: {repr(line)}")
        else:
            print("❌ Failed to extract text")
        
        print("-" * 40)

if __name__ == "__main__":
    test_title_extraction()
