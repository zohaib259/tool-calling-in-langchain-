from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


# 1. Define the function with type hints and a docstring
@tool
def multiply(a: int, b: int) -> int:
  """Multiply two numbers together and return the result."""
  return a * b


# 2. Test the tool directly
result = multiply.invoke({"a": 4, "b": 5})


llm = ChatGoogleGenerativeAI(model = "gemini-3.8-flash")
llm_with_tools = llm.bind_tools([multiply])

response = llm_with_tools.invoke("What is 12 multiplied by 8?")

print(response.tool_calls)