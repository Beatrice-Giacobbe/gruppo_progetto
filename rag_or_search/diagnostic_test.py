#!/usr/bin/env python3
"""
Test diretto del modulo RAG
"""

import os
import sys

# Aggiungi il percorso corretto al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_environment():
    """Test delle variabili di ambiente"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'LMSTUDIO_MODEL']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ Variabile mancante: {var}")
        else:
            print(f"✅ {var}: {'*' * min(10, len(value))}...")
    
    return len(missing_vars) == 0

def test_pdf_file():
    """Test presenza file PDF"""
    pdf_path = os.path.join(current_dir, 'src', 'rag_or_search', 'tools', 'RAG_qdrant_new', 'knowledge_base', 'EU AI Act.pdf')
    if os.path.exists(pdf_path):
        print(f"✅ File PDF trovato: {pdf_path}")
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        print(f"   Dimensione: {file_size:.2f} MB")
        return True
    else:
        print(f"❌ File PDF non trovato: {pdf_path}")
        return False

def test_qdrant_connection():
    """Test connessione a Qdrant"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=5)
        
        # Test connessione
        collections = client.get_collections()
        print(f"✅ Connessione Qdrant riuscita")
        print(f"   Collections esistenti: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"❌ Errore connessione Qdrant: {e}")
        return False

def test_imports():
    """Test importazioni critiche"""
    try:
        import langchain
        import langchain_openai
        import qdrant_client
        import pdfminer
        from dotenv import load_dotenv
        print("✅ Tutte le importazioni riuscite")
        return True
    except ImportError as e:
        print(f"❌ Errore importazione: {e}")
        return False

def main():
    print("=== DIAGNOSTICA SISTEMA RAG ===\n")
    
    all_tests_passed = True
    
    print("1. Test importazioni:")
    all_tests_passed &= test_imports()
    
    print("\n2. Test variabili di ambiente:")
    all_tests_passed &= test_environment()
    
    print("\n3. Test file PDF:")
    all_tests_passed &= test_pdf_file()
    
    print("\n4. Test connessione Qdrant:")
    all_tests_passed &= test_qdrant_connection()
    
    print("\n" + "="*50)
    
    if all_tests_passed:
        print("✅ TUTTI I TEST PASSATI - Il sistema è pronto!")
        print("\nPer testare il RAG, compila il file .env con la tua OPENAI_API_KEY")
        print("e poi esegui il test completo.")
    else:
        print("❌ ALCUNI TEST FALLITI - Controlla i problemi sopra")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
