from random import randint
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from evaluation_flow.crews.eval_crew.eval_crew import ResearchCrew

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


if __name__ == "__main__":
    kickoff()
    plot()
