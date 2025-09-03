from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from typing import List
from rag_qdrant_hybrid import CURRENT_DIRECTORY_PATH, SETTINGS, format_docs_for_prompt, get_embeddings, get_llm, get_qdrant_client, hybrid_search, load_pdf, recreate_collection_for_rag, retry_with_backoff, split_documents, upsert_chunks, build_rag_chain
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)

    # REMOVED: get_contexts_for_question is obsolete with new build_ragas_dataset

def build_ragas_dataset(
    questions: List[str],
    client,
    s,
    embeddings,
    hybrid_search,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
    """
    dataset = []
    chain = build_rag_chain(get_llm(s))
    for q in questions:
        docs = hybrid_search(client, s, q, embeddings)
        contexts = [doc.payload["text"] if hasattr(doc, "payload") and "text" in doc.payload else getattr(doc, "page_content", str(doc)) for doc in docs]
        formatted_context = format_docs_for_prompt(docs)
        answer = chain.invoke({"question": q, "context": formatted_context})

        row = {
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


if __name__ == "__main__":
    s = SETTINGS
    embeddings = get_embeddings(s)
    llm = get_llm(s)
    client = get_qdrant_client(s)
    # docs = simulate_corpus()
    # loader = DirectoryLoader(f"{CURRENT_DIRECTORY_PATH}/../../../../outputs", glob="ai_act.md")
    # docs = loader.load()
    docs = load_pdf(f"{CURRENT_DIRECTORY_PATH}/knowledge_base/EU AI Act.pdf")
    chunks = split_documents(docs, s)
    print(f"Docs: {len(docs)}, Chunks: {len(chunks)}")
    
    # Use retry logic for initial embedding call
    def get_vector_size():
        return len(embeddings.embed_query("hello world"))
    
    vector_size = retry_with_backoff(get_vector_size, max_retries=5, base_delay=2.0)
    recreate_collection_for_rag(client, s, vector_size)
    
    # You only need to run upsert_chunks if your collection is empty or you want to update it.
    # If the chunks are already embedded and present in Qdrant, you can skip this step.
    if not client.count(collection_name=s.collection).count:
        upsert_chunks(client, s, chunks, embeddings)
    else:
        print("Collection already populated, skipping upsert.")

    # 5) Esempi di domande
    questions = [
        "Cosa deve fare un fornitore di un sistema AI ad alto rischio prima di metterlo sul mercato in UE?",
        "Chi è responsabile della valutazione dei rischi per un sistema AI ad alto rischio?",
        "I sistemi AI ad alto rischio devono rispettare specifici obblighi di trasparenza. Quale di questi è corretto?",
    ]

    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        ans = hybrid_search(client, s, q, embeddings)
        print(ans)
        print()

    # (opzionale) ground truth sintetica per correctness
    ground_truth = {
        questions[0]: "Deve garantire che il sistema AI sia conforme ai requisiti dell’AI Act, che includono: Eseguire una Valutazione della Conformità (Conformity Assessment) per verificare che il sistema soddisfi gli obblighi di sicurezza, trasparenza, accuratezza, gestione dei dati e documentazione. Redigere e conservare una dichiarazione di conformità UE. Creare e mantenere documentazione tecnica completa del sistema AI. Implementare misure per tracciabilità e monitoraggio continuo del sistema AI. Assicurarsi che il sistema sia adeguatamente etichette e istruzioni per l’uso, con indicazioni sui limiti e rischi.",
        questions[1]: "Deve identificare, analizzare e mitigare i rischi potenziali legati a sicurezza, protezione dei dati e diritti fondamentali. La valutazione deve essere documentata nella Valutazione dei Rischi e parte della documentazione tecnica.",
        questions[2]: "I sistemi ad alto rischio devono fornire informazioni chiare sull’uso dell’AI, sulle capacità e sui limiti. Gli altri punti (codice open-source o accesso gratuito ai dati) non sono obbligatori secondo l’AI Act.",
    }

    # 6) Costruisci dataset per Ragas (stessi top-k del tuo retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        client=client,
        s=s,
        embeddings=embeddings,
        hybrid_search=hybrid_search,
        k=s.final_k,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # 7) Scegli le metriche
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=get_embeddings(s),  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")
    