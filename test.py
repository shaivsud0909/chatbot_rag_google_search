from langchain_core.messages import HumanMessage
from app import chatbot   

THREAD_ID = "1"   

print("Chatbot started. Type 'exit' to stop.\n")

while True:
    user_message = input("You: ").strip()

    if user_message.lower() in ["exit", "quit", "bye"]:
        print("Chat ended.")
        break

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }

    result = chatbot.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )

    # Get last AI message safely
    ai_message = result["messages"][-1]

    # Gemini usually returns content as list[dict]
    if isinstance(ai_message.content, list):
        print("AI:", ai_message.content[0].get("text", ""))
    else:
        print("AI:", ai_message.content)
