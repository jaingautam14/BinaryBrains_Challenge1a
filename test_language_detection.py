#!/usr/bin/env python3
"""
Test script for language detection functionality
Demonstrates language detection with various text samples
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_pdfs import PDFProcessor

def test_language_detection():
    """Test language detection with sample texts"""
    processor = PDFProcessor()
    
    # Test samples in different languages
    test_samples = [
        {
            "name": "English",
            "text": "This is a comprehensive PDF processing solution that extracts structured data from PDF documents. The system uses advanced OCR technology to handle both text-based and image-based PDF files.",
            "expected": "en"
        },
        {
            "name": "French",
            "text": "Ceci est un rapport annuel complet qui présente les résultats financiers de notre société. L'analyse comprend les revenus, les profits et les investissements stratégiques pour l'année écoulée.",
            "expected": "fr"
        },
        {
            "name": "Spanish",
            "text": "Este documento presenta una propuesta técnica para el desarrollo de un sistema de procesamiento de documentos. La solución incluye capacidades avanzadas de reconocimiento óptico de caracteres.",
            "expected": "es"
        },
        {
            "name": "German",
            "text": "Dieses Dokument beschreibt die technischen Spezifikationen für ein neues Dokumentenverarbeitungssystem. Die Lösung unterstützt verschiedene PDF-Formate und bietet erweiterte Texterkennungsfunktionen.",
            "expected": "de"
        },
        {
            "name": "Mixed English-French",
            "text": "This document contains both English and French text. Nous présentons ici une solution bilingue qui peut traiter des documents dans plusieurs langues simultanement.",
            "expected": "en"  # Should detect English as primary
        }
    ]
    
    print("Language Detection Test Results:")
    print("=" * 50)
    
    for sample in test_samples:
        # Create mock pages_text structure
        pages_text = [{"text": sample["text"], "page_num": 0}]
        
        # Detect language
        result = processor.detect_language(pages_text)
        
        # Display results
        print(f"\nTest: {sample['name']}")
        print(f"Expected: {sample['expected']}")
        print(f"Detected: {result['primary']} (confidence: {result['confidence']:.3f})")
        
        if result['additional']:
            print("Additional languages:")
            for lang in result['additional']:
                print(f"  - {lang['language']}: {lang['confidence']:.3f}")
        
        # Check if detection is correct
        status = "✅ PASS" if result['primary'] == sample['expected'] else "❌ FAIL"
        print(f"Status: {status}")
        print("-" * 30)

if __name__ == "__main__":
    test_language_detection()
