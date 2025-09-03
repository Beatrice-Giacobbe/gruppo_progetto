#!/usr/bin/env python3
"""
Test script per verificare il funzionamento del RAG pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_or_search.tools.RAG_qdrant_new.rag_qdrant_hybrid import search_rag

def test_rag_pipeline():
    """Test del pipeline RAG"""
    try:
        print("=== Test RAG Pipeline ===")
        
        # Domanda di test
        question = "Cosa dice l'AI Act europea sui sistemi di intelligenza artificiale ad alto rischio?"
        k = 3  # Numero di risultati da restituire
        
        print(f"Domanda: {question}")
        print(f"Numero di chunk richiesti: {k}")
        print("\n" + "="*50)
        
        # Esegui la ricerca
        result = search_rag(question, k)
        
        print("\n=== RISULTATI ===")
        print(result)
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_imports():
    """Test di importazione delle dipendenze"""
    try:
        print("=== Test Importazioni ===")
        
        import qdrant_client
        print("✅ qdrant_client importato")
        
        import langchain
        print("✅ langchain importato")
        
        import langchain_openai
        print("✅ langchain_openai importato")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv importato")
        
        import pdfminer
        print("✅ pdfminer.six importato")
        
        return True
        
    except ImportError as e:
        print(f"❌ Errore di importazione: {e}")
        return False

if __name__ == "__main__":
    print("Avvio test del sistema RAG...")
    
    # Test delle importazioni
    if not test_basic_imports():
        print("❌ Test delle importazioni fallito")
        sys.exit(1)
    
    print("\n" + "="*50)
    
    # Test del pipeline RAG
    if test_rag_pipeline():
        print("\n✅ Test completato con successo!")
    else:
        print("\n❌ Test fallito!")
        sys.exit(1)
