from typing import Type
from crewai.tools import tool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
#from duckduckgo_search import DDGS
from ddgs import DDGS


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

@tool
def cerca_ddg(topic: str, n: int = 3) -> list[dict]:
    """Cerca su DuckDuckGo e restituisce i primi n risultati come lista di dict."""
    with DDGS(verify=False) as ddgs:
        return [
            {
                "title": r.get("title"),
                "snippet": r.get("body")
            }
            for r in ddgs.text(topic, region="it-it", safesearch="off", max_results=n)
        ]