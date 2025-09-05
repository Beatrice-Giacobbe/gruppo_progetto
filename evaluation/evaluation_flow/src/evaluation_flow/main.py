import os
from random import randint
#from evaluation_flow.tools.custom_tool import cerca_ddg
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from evaluation_flow.crews.eval_crew.eval_crew import ResearchCrew
from deepeval.integrations.crewai import instrument_crewai, Agent
import json
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import EvaluationDataset, Golden
from crewai import LLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import AzureOpenAIModel
from deepeval import evaluate
from deepeval import login
login("confident_us_tC1XmkQJRc25Zphtv5Ws0oGKLYA3LTczm60FNiatE+Vsikqk8=")
instrument_crewai()
load_dotenv()


class ResearchCrewState(BaseModel):
    #sentence_count: int = 1
    topic: str = ""

class ResearchCrewFlow(Flow[ResearchCrewState]):
    """A flow to search on duckduck with an input query."""    
    @start()
    def ask_query(self):
        print("Asking query")
        self.state.topic = input("Inserisci l'argomento da ricercare: ").strip()
        return self.state.topic


    @listen(ask_query)
    def make_search(self):
        result = ResearchCrew().crew().kickoff(inputs={'topic': self.state.topic})

        # Print the result
        print("\n\n=== FINAL REPORT ===\n\n")
        print(result.raw)
        print("\n\nReport has been saved to output/risultato.txt")


def kickoff():
    research_flow = ResearchCrewFlow()
    research_flow.kickoff()

def plot():
    research_flow = ResearchCrewFlow()
    research_flow.plot("plot flow.html")

def run_evaluation():
    test_cases = []
    golden_dataset = [
    {
        "question": "Cos'è un sistema RAG e a cosa serve?",
        "expected_answer": "Un sistema Retrieval-Augmented Generation combina un motore di retrieval di documenti con un modello generativo: prima recupera passaggi rilevanti dal corpus e poi il modello genera la risposta citando/riassumendo tali passaggi, migliorando factualità e copertura."
    },
    {
        "question": "Qual è la differenza tra precision e recall?",
        "expected_answer": "La precision è la proporzione di predizioni positive corrette sul totale delle predizioni positive; il recall è la proporzione di positivi correttamente individuati sul totale dei positivi reali."
    },
    {
        "question": "Perché si usano gli embedding in un RAG?",
        "expected_answer": "Gli embedding trasformano testi in vettori numerici che catturano la semantica, permettendo di calcolare similarità e recuperare documenti pertinenti in base al significato e non solo a parole esatte."
    },
    {
        "question": "Cosa fa la normalizzazione dei valori VAD?",
        "expected_answer": "Riporta le scale di Valence, Arousal e Dominance su un intervallo comune per renderle confrontabili, stabilizzare l'addestramento e ridurre l'influenza di range diversi."
    },
    {
        "question": "Quando conviene usare MMR nel retrieval?",
        "expected_answer": "Quando vuoi bilanciare rilevanza e diversità tra i documenti recuperati, evitando risultati ridondanti e coprendo aspetti diversi della query."
    },
    {
        "question": "Differenza tra z-score e IQR per outlier?",
        "expected_answer": "Lo z-score misura quanto un punto è distante in deviazioni standard dalla media, l'IQR usa i quantili Q1 e Q3 e considera outlier i punti oltre 1.5×IQR dai quartili; IQR è più robusto a distribuzioni non gaussiane."
    },
    {
        "question": "A cosa serve Docker in un progetto ML?",
        "expected_answer": "A rendere riproducibile l'ambiente di esecuzione impacchettando dipendenze e configurazioni in un container portabile tra macchine e server."
    },
    {
        "question": "Che cos'è Qdrant in una pipeline RAG?",
        "expected_answer": "È un database vettoriale che indicizza embedding e consente ricerche di similarità ad alte prestazioni, con filtri e funzioni come MMR e payloads."
    },
    {
        "question": "Perché separare data aggregation, cleaning e transformation?",
        "expected_answer": "Per ottenere modularità, riuso e tracciabilità: ogni fase ha responsabilità chiare, facilita il debug e permette di scalare/ri-eseguire step specifici."
    },
    {
        "question": "Cosa misura la metrica BLEU a grandi linee?",
        "expected_answer": "Misura la sovrapposizione di n-gram tra testo generato e riferimento, penalizzando frasi troppo corte; è usata per valutare traduzione o generazione testuale."
    }]
    goldens = []
    for data in golden_dataset:
        golden = Golden(
            input = data["question"],
            expected_output = data["expected_answer"]
        )
        goldens.append(golden)
 
    dataset = EvaluationDataset(goldens)

# istanzia la crew definita altrove
    crew = ResearchCrew().crew()  
    #dataset = EvaluationDataset(goldens=[Golden(input="Explain the latest trends in AI.")])
    
    model_azure = AzureOpenAIModel(
        model_name="gpt-4o",
        deployment_name=os.getenv("OPENAI_MODEL_NAME"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
        openai_api_version=os.getenv("AZURE_API_VERSION1"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1"),
        temperature=0
    )

    answer_relevancy_metric = AnswerRelevancyMetric(model = model_azure)
    for golden in dataset.evals_iterator(): #or dataset.goldens
        result = crew.kickoff(inputs={'topic': golden.input})

        # Se result è un percorso a file .md
        if isinstance(result, str) and result.endswith(".md"):
            with open(result, "r", encoding="utf-8") as f:
                result_str = f.read()
        else:
            # Altrimenti è già una stringa o un oggetto convertibile
            result_str = str(result)
        test_cases.append(
            LLMTestCase(
            input=golden.input,
            actual_output=result_str
            )
        )

    evaluate(test_cases=test_cases, metrics=[answer_relevancy_metric])

"""
    for golden in dataset.evals_iterator():
        # Avvia la tua crew con l’input del golden
        result = crew.kickoff(inputs={'topic': golden.input})
    
        # Crea il test case per Deepeval
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=result,
            expected_output=golden.expected_output
        )
        test_cases.append(test_case)
        #dataset.add(test_case)

    # Valutazione automatica
    evaluate(test_cases)"""


    
if __name__ == "__main__":
    #kickoff()
    #plot()
    run_evaluation()
