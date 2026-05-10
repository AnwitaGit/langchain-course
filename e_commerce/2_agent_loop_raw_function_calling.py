from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


import ollama

MAX_ITERATIONS = 10 
MODEL = "qwen3:1.7b"


#Tools
@traceable(run_type="tool")
def get_product_price(product:str)->float:
    "Lookup the price of product in catalog"
    prices= {"laptop":1299.99, "headphones":149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price:float, discount_tier:str) ->float : 
    "apply discount to the price of a product based on discount_tier"
    """ Available tiers: Bronze, Silver, Gold"""
    discount_percentages = {"bronze": 5, "silver": 12, "gold":23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price* ((100-discount)/100), 2)

tools_for_llm = [
    {
        "type": "function",
        "function":{
            "name":"get_product_price",
            "description":"Look up the price of a product in the catalog.",
            "parameters":
             {
                 "type":"object",
                 "properties":{
                     "product":{
                         "type": "string",
                         "description": "The product name. Ex: laptop, headphones, keyboard.",
                     }
                 },
                 "required": ["product"]
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
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },

]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
#Agent--loop

@traceable(name = "Ollama Agent Loop")
def run_agent(question:str):

    tools_dict = {"get_product_price": get_product_price,
                  "apply_discount": apply_discount
                  
                  }

    messages = [
        {
            "role": "system",
            "content":
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool an a discount tool."
                "STRICT RULES - you must follow these eactly:\n"
                "1. Never assume or guess a product price."
                "You must call get_prouct_price first to get the product price. \n"
                "2. Just call apply discount after getting the product price."
                "3. Never calculate discounts yoursel using math"
                " Always use the apply_discount tool"
                "4. If the user does not specify a discount tier, ask then which teir to use - do not assme one."
        },
        
        {"role": "user", "content": question},

    ]
    for iteration in range(1, MAX_ITERATIONS+1):
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        if not tool_calls: 
            print(f"Final answer: {ai_message}")
            return ai_message.content


        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name}  not found")
        observation = tool_to_use(**tool_args)

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool", 
                "content": str(observation)
            }

        )

    

if __name__ =="__main__":
    print("Hello Langchain Agent (.bind tools)")
    print()
    result = run_agent("What is the price of headphones after applying a silver discount?")