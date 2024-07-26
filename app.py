import os
from typing import Iterator
import gradio as gr
from model import get_input_token_length, get_LLAMA_response_stream, get_LLAMA_response
import datetime
import chromadb

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
DEFAULT_SYSTEM_PROMPT = """あなたは親切で楽しい精神療法助手です。安全でありながら、できるだけ親切かつ元気に答えてください。あなたの答えは、有害、非倫理的、人種差別主義者、性差別主義者、毒性、危険、または違法な内容を含めるべきではありません。あなたの回答が社会的に偏りがなく、本質的に前向きであることを確認してください。質問が意味をなさない場合、または事実上一貫性がない場合は、正しいものに答えるのではなく、理由を説明してください。質問に対する答えがわからない場合は、虚偽の情報を共有しないでください。"""

DESCRIPTION = """
# Arisara-Mental-Therapy-Chatbot
"""
LICENSE = "open-source"

#Setup DB locally
client = chromadb.PersistentClient(path="chat_history") #set persistent client path
collection = client.get_or_create_collection("chat_history") #set collection's name


def add_chat_history(conversation):
  timestamp = datetime.datetime.utcnow()
  timestamp=str(timestamp)
  collection.add(
    documents = [f"{conversation}"],
    ids = [f"{timestamp}"])
  print('save history')


def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50
) -> Iterator[str]:
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        for user, assistant in chat_history:
            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        conversation.append({"role": "user", "content": message})

        
        if(get_input_token_length(conversation) > MAX_INPUT_TOKEN_LENGTH):
            raise gr.InterfaceError(f"The accumulated input is too long ({get_input_token_length(conversation)} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.")
        
        #Altering use between 'get_LLAMA_response_stream' and 'get_LLAMA_response'
        #In this case, responses will be a streaming for asynchronous manner and continuously real-time interaction
        #Otherwise, response will be a single complete response generated from input messages at once. 
        generator = get_LLAMA_response_stream(conversation, max_new_tokens, temperature, top_p, top_k)
        
        result = ''
        for response in generator:
           yield response
           result+=response
        add_chat_history(f'{message}->{result}') #check point if chat history is saved to database
        print('yield!') #check point if response is generated 


        
        

chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label='System Prompt',
                                   value=DEFAULT_SYSTEM_PROMPT,
                                   lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=1,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
    ],
    stop_btn="Stop",
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()
if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
