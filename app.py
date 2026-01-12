from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from tools import tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,tools_condition

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=api_key
)
 
#tool binding
llm=llm.bind_tools(tools)
tool_node=ToolNode(tools)

#state
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# conn=sqlite3.connect(database='chatbot.db')
# checkpointer=SqliteSaver(conn=conn)

checkpointer=MemorySaver()

graph=StateGraph(ChatState)   

graph.add_node('chat_node',chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools", "chat_node") 
graph.add_edge('chat_node',END)



chatbot=graph.compile(checkpointer=checkpointer)

