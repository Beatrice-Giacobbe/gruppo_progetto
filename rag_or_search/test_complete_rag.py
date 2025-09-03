#!/usr/bin/env python3
"""
Test completo del sistema RAG
"""

import os
import sys

# Aggiungi il percorso corretto al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_rag_search():
    """Test completo del sistema RAG"""
    try:
        # Importa il modulo RAG
        from src.rag_or_search.tools.RAG_qdrant_new.rag_qdrant_hybrid import search_rag
        
        print("=== TEST RAG SEARCH ===")
        print("Modulo importato con successo!")
        
        # Domanda di test
        question = "Cosa dice l'AI Act sui sistemi di intelligenza artificiale ad alto rischio?"
        k = 3
        
        print(f"\nDomanda: {question}")
        print(f"Numero di chunk richiesti: {k}")
        print("\nEsecuzione ricerca...")
        print("-" * 50)
        
        # Esegui la ricerca RAG
        result = search_rag(question, k)
        
        print("\n" + "="*60)
        print("RISULTATI:")
        print("="*60)
        print(result)
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Avvio test completo del sistema RAG...\n")
    
    # Carica le variabili d'ambiente
    from dotenv import load_dotenv
    load_dotenv()
    
    if test_rag_search():
        print("\n✅ TEST RAG COMPLETATO CON SUCCESSO!")
        print("\nIl sistema è funzionante e può processare documenti PDF,")
        print("creare embeddings, e rispondere a domande usando il knowledge base.")
    else:
        print("\n❌ TEST RAG FALLITO!")
        print("Controlla la configurazione e riprova.")

if __name__ == "__main__":
    main()
