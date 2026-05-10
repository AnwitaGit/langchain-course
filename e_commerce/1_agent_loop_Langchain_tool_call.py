from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10 
MODEL = "qwen3:1.7b"


#Tools
@tool
def get_product_price(product:str)->float:
    "Lookup the price of product in catalog"
    prices= {"laptop":1299.99, "headphones":149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price:float, discount_tier:str) ->float : 
    "apply discount to the price of a product based on discount_tier"
    """ Available tiers: Bronze, Silver, Gold"""
    discount_percentages = {"bronze": 5, "silver": 12, "gold":23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price* ((100-discount)/100), 2)

#Agent--loop

@traceable(name = "Langchain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(model="gpt-5", model_provider="openai", temperature=0)

    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content = (
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool an a discount tool."
                "STRICT RULES - you must follow these eactly:\n"
                "1. Never assume or guess a product price."
                "You must call get_prouct_price first to get the product price. \n"
                "2. Just call apply discount after getting the product price."
                "3. Never calculate discounts yoursel using math"
                " Always use the apply_discount tool"
                "4. If the user does not specify a discount tier, ask then which teir to use - do not assme one."
            )
        ),
        HumanMessage(content=question),

    ]
    for iteration in range(1, MAX_ITERATIONS+1):
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls: 
            print(f"Final answer: {ai_message}")
            return ai_message.content


        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name}  not found")
        observation = tool_to_use.invoke(tool_args)

        messages.append(ai_message)
        messages.append(
            ToolMessage(content = str(observation), tool_call_id=tool_call_id)

        )

    

if __name__ =="__main__":
    print("Hello Langchain Agent (.bind tools)")
    print()
    result = run_agent("What is the price of headphones after applying a silver discount?")