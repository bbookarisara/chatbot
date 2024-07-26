from threading import Thread
from typing import Iterator, List, Dict
from transformers import (AutoTokenizer,AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig)
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM


# Configurations for 4-bit quantization

MAX_INPUT_TOKEN_LENGTH = 4096

use_4bit = True
device_map = {"": 0}  # Ensures model is loaded on the GPU
device = 'cuda:0'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

#set up model

model_name = "bbookarisara/arisara_llama3_for_mental_therapy"
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config,device_map=device_map)


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_fast=False,add_eos_token=True)
tokenizer.pad_token_id = 18610

# Function to get input token length
def get_input_token_length(messages) -> int:
    return len(tokenizer.apply_chat_template(messages))

# Function to get response stream
def get_LLAMA_response_stream(
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> Iterator[str]:
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')

    
    if len(inputs["input_ids"]) > MAX_INPUT_TOKEN_LENGTH:
        raise ValueError(f"Input token length is {inputs['input_ids'].shape[1]}, which exceeds the maximum of {MAX_INPUT_TOKEN_LENGTH}.")
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)

#Function to get response string
def get_LLAMA_response(
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> str:
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    
    if len(input_ids) > MAX_INPUT_TOKEN_LENGTH:
        raise ValueError(f"Input token length is {inputs['input_ids'].shape[1]}, which exceeds the maximum of {MAX_INPUT_TOKEN_LENGTH}.")
    
    output_ids = model.generate(
        **inputs,
        max_length=4096,  # sum of input_tokens + max_new_tokens
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )
    output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text
