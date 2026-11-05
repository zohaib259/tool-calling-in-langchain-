from langchain_core.tools import tool


# 1. Define the function with type hints and a docstring
@tool
def multiply(a: int, b: int) -> int:
  """Multiply two numbers together and return the result."""
  return a * b


# 2. Test the tool directly
result = multiply.invoke({"a": 4, "b": 5})
print(result)  # Output: 20