# !pip install langchain langchain-community
import os
from langchain_community.llms.sambanova import SambaStudio

# Get credentials from environment variables
SAMBASTUDIO_URL = os.getenv('SAMBASTUDIO_URL')
SAMBASTUDIO_API_KEY = os.getenv('SAMBASTUDIO_API_KEY')

if not SAMBASTUDIO_URL or not SAMBASTUDIO_API_KEY:
    print("❌ Error: Please set SAMBASTUDIO_URL and SAMBASTUDIO_API_KEY environment variables")
    print("   Run: source ~/.bashrc")
    print("   Or set them manually:")
    print("   export SAMBASTUDIO_URL=your_url")
    print("   export SAMBASTUDIO_API_KEY=your_key")
    exit(1)

model = "Meta-Llama-3-70B-Instruct" # "Meta-Llama-3-8B-Instruct" # "Meta-Llama-3-70B-Instruct"

def call_llm_fn(prompt:str, model=model, max_tokens=1024):
        llama = SambaStudio(
            model_kwargs={
                "do_sample":True,
                "max_tokens":max_tokens,
                "temperature":0.01,
                "process_prompt":False,
                "model":model,
            },
        )
           
        res = llama.invoke(prompt)
        return res


res = call_llm_fn(prompt="What is the capital of France?", model=model)
print(res)
