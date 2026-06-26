from dotenv import load_dotenv


load_dotenv()


from langsmith import traceable
import ollama

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# --- Tools (LangChain @tool decorator) ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f".   >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"   >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools_for_llm = [
    {
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Look up the price of a product in the catalog",
        "parameters": {
          "type": "object",
          "required": ["product"],
          "properties": {
            "product": {"type": "string", "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'"}
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "apply_discount",
        "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
        "parameters": {
          "type": "object",
          "required": ["price", "discount_tier"],
          "properties": {
            "price": {"type": "number", "description": "The original price"},
            "discount_tier": {"type": "string", "description": "The discount tier: 'bronze', 'silver', or 'gold'"}
          }
        }
      }
    }
]


# Without langchain we must manually trace ollama calls to langsmith
@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ---- AGENT LOOP ----
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "apply_discount": apply_discount,
        "get_product_price": get_product_price
    }
    

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {"role": "system", "content": (
            "You are a helpful shopping assistant. "
            "You have access to a product catalog tool "
            "and a discount tool. \n\n"
            "STRICT RULES - you must follow these exactly:\n"
            "1. NEVER guess or assume any product price. "
            "You MUST call get_product_price first to get the real price.\n"
            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NOT pass a made-up number.\n"
            "check if the question has a product name\n"
            "users already provided the product name in the question, do not ask further information"
        )},
        {"role": "human", "content": question}
    ]


    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
    
        response = ollama_chat_traced(messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # if not tool calls this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call for convenience
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if (tool_to_use) is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use(**tool_args)

        print(f"   [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )
    
    print("ERROR: Max iterations reached without a final answer")
    return None

                                  


def main():
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a 'laptop' after applying a gold discount?")


if __name__ == "__main__":
    main()
