
import re
import inspect 
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
    price = float(price)
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price* ((100-discount)/100), 2)

tools = {"get_product_price": get_product_price, 
     "apply_discount" : apply_discount
    }

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
        return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt =f"""STRICT RULES - you must follow these eactly:\n"
"1. Never assume or guess a product price."
"You must call get_prouct_price first to get the product price. \n"
"2. Just call apply discount after getting the product price."
"3. Never calculate discounts yoursel using math"
" Always use the apply_discount tool"
"4. If the user does not specify a discount tier, ask then which teir to use - do not assme one.
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options = options)
#Agent--loop

@traceable(name = "Ollama Agent Loop")
def run_agent(question:str):
    prompt = react_prompt.format(question = question)
    scratchpad = ""


    for iteration in range(1, MAX_ITERATIONS+1):
        full_prompt = prompt + scratchpad
        response = ollama_chat_traced(
            model = MODEL, 
            messages = [{"role": "user", "content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature":0})
        output = response.message.content
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match: 
            final_answer = final_answer_match.group(1).strip()
            return final_answer
        

        action_match = re.search(r"Action:\s(.+)", output)
        action_input_match = re.search(r"Action Input Match: \s(.+)", output)

        if not action_match or not action_input_match:
            print(" [Parsing] Error: Could not find a match for action or action input")
            break
        
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\'") for x in raw_args]
        if tool_name not in tools: 
            observation = f"Error: Tool '{tool_name}' not found "
        else: 
            observation = str(tools[tool_name](*args))
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    

if __name__ =="__main__":
    print("Hello Langchain Agent (.bind tools)")
    print()
    result = run_agent("What is the price of headphones after applying a silver discount?")