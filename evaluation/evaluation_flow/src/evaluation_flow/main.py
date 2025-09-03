from random import randint
from evaluation_flow.tools.custom_tool import cerca_ddg
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
from deepeval import evaluate
instrument_crewai()

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

def evaluate():
    test_cases = []
 
# istanzia la crew definita altrove
    crew = ResearchCrew().crew()  
    dataset = EvaluationDataset(goldens=[Golden(input="Explain the latest trends in AI.")])
    answer_relevancy_metric = AnswerRelevancyMetric()
    for golden in dataset.goldens:
        dataset.add_test_case(
            LLMTestCase(
            input=golden.input,
            actual_output=crew.kickoff(inputs={'topic': golden.input})
            )
        )
 
    evaluate(test_cases=dataset.test_cases, metrics=[answer_relevancy_metric])

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
    evaluate()
