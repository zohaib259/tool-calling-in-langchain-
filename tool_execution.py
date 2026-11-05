from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
  """Multiply two numbers together and return the result."""
  return a * b


query = HumanMessage("What is 12 multiplied by 8?")
messages = [query]

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
llm_with_tools = llm.bind_tools([multiply])

response = llm_with_tools.invoke(query)
messages.append(response)

tool_call = response.tool_calls[0]
tool_output = multiply.invoke(tool_call)
tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
messages.append(tool_message)

final_response = llm_with_tools.invoke(messages)
print(final_response.text)  # Output: 96