from langchain.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
from rag import rag_search

load_dotenv()


def google_search(querry:str)->str:
    """search google and return the top result """
    #connect
    service = build(
        "customsearch",
        "v1",
        developerKey=os.getenv("GOOGLE_API_KEY")
    )
    #choose engine
    res = service.cse().list(
        q=querry,
        cx=os.getenv("GOOGLE_CSE_ID"),
        num=5
    ).execute() #send request

    results = []
    for item in res.get("items", []):
        results.append(f"{item['title']}: {item['link']}")
    
    return "\n".join(results)


tools=[google_search,rag_search]