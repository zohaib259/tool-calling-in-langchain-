from typing import Annotated
import json
import requests
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() 

# 1. First Tool: Find the current exchange rate
@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> float:
    """Get the current live exchange rate from a base currency to a target currency (e.g., USD to EUR)."""
    url = f"https://open.er-api.com/v6/latest/{base_currency.upper()}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("result") == "success":
            rates = data.get("rates", {})
            rate = rates.get(target_currency.upper())
            if rate:
                return float(rate)
        return 0.0
    except Exception as e:
        return 0.0


# 2. Second Tool: Multiply the base amount by the exchange rate
@tool
def multiply_currency(amount: float, exchange_rate: Annotated[float, InjectedToolArg]) -> float:
    """Multiply the base currency amount by the fetched exchange rate to get the final converted amount."""
    return amount * exchange_rate


# Initialize the Gemini model and bind both tools
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
llm_with_tools = llm.bind_tools([get_exchange_rate, multiply_currency])

# Set up the conversation query
messages = [HumanMessage(content="I want to convert 1500 USD to pkr. What is the final amount?")]

# --- TURN 1: Model triggers tool calls ---
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

print("--- Turn 1 Tool Calls ---")
print(ai_message.tool_calls)

exchange_rate = None

# Iterate through the tool calls using your preferred loop structure
for tool_call in ai_message.tool_calls:
    # Execute the 1st tool and get the value of exchange rate
    if tool_call['name'] == 'get_exchange_rate':
        # Invoke tool and wrap result in a ToolMessage
        rate_result = get_exchange_rate.invoke(tool_call)
        
        # If your tool returns a direct float, convert/wrap it cleanly
        exchange_rate = float(rate_result)
        
        tool_message1 = ToolMessage(
            content=json.dumps({"exchange_rate": exchange_rate}), 
            tool_call_id=tool_call['id']
        )
        
        # Append this tool message to messages list
        messages.append(tool_message1)

    # Execute the 2nd tool using the exchange rate from tool 1
    if tool_call['name'] == 'multiply_currency':
        # Fetch and inject the current arg
        tool_call['args']['exchange_rate'] = exchange_rate
        
        final_amount = multiply_currency.invoke(tool_call)
        
        tool_message2 = ToolMessage(
            content=str(final_amount), 
            tool_call_id=tool_call['id']
        )
        
        messages.append(tool_message2)

# --- FINAL TURN: Model generates the final natural language answer ---
final_response = llm_with_tools.invoke(messages)
print("\n--- Final Answer ---")
print(final_response.content)