# -*- coding: utf-8 -*-
"""
Showing how a prompt using the superbooga extension can work
"""

import requests

HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

def generate_text(prompt,instructions= "Analyze the provided documents and websites to find relevant information related to my questions, and summarize it in a clear and concise manner.", max_new_tokens=250, do_sample=True, temperature=1.3, top_p=0.1,
                  typical_p=1, repetition_penalty=1.18, top_k=40, min_length=0, no_repeat_ngram_size=0,
                  num_beams=1, penalty_alpha=0, length_penalty=1, early_stopping=False, seed=-1,
                  add_bos_token=True, truncation_length=2048, ban_eos_token=False, skip_special_tokens=True,
                  stopping_strings=[]):
    """
    Generates text using OpenAI's Codex API based on the given prompt and configuration parameters.
    
    Args:
        prompt (str): The user prompt to use as input for text generation.
        instructions (str): The instructions to use for the generated text.
        max_new_tokens (int): The maximum number of tokens to generate in the output text.
        do_sample (bool): Whether to use sampling instead of greedy decoding.
        temperature (float): The temperature to use in the softmax temperature scaling.
        top_p (float): The cumulative probability threshold to use for nucleus sampling.
        typical_p (float): The typical probability of the next token when using nucleus sampling.
        repetition_penalty (float): The penalty to apply to repeated tokens.
        top_k (int): The number of top tokens to consider when using top-k sampling.
        min_length (int): The minimum length of the generated text.
        no_repeat_ngram_size (int): The size of the n-gram window to avoid repetition.
        num_beams (int): The number of beams to use for beam search.
        penalty_alpha (float): The length penalty exponent to use for beam search.
        length_penalty (float): The length penalty to use for beam search.
        early_stopping (bool): Whether to stop generation when all beams have finished.
        seed (int): The random seed to use for generation.
        add_bos_token (bool): Whether to add the beginning-of-sequence token to the generated text.
        truncation_length (int): The maximum length of the generated text.
        ban_eos_token (bool): Whether to ban the end-of-sequence token from the generated text.
        skip_special_tokens (bool): Whether to skip special tokens in the generated text.
        stopping_strings (List[str]): A list of strings that, when encountered in the generated text,
            cause generation to stop.
    
    Returns:
        str: The generated text.
    """
    formatted_prompt = f"""Instruction: {instructions}

<|injection-point|>

Input:
<|begin-user-input|>
{prompt}?

<|end-user-input|>
Response:
"""
    
    request = {
        'prompt': formatted_prompt,
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'temperature': temperature,
        'top_p': top_p,
        'typical_p': typical_p,
        'repetition_penalty': repetition_penalty,
        'top_k': top_k,
        'min_length': min_length,
        'no_repeat_ngram_size': no_repeat_ngram_size,
        'num_beams': num_beams,
        'penalty_alpha': penalty_alpha,
        'length_penalty': length_penalty,
        'early_stopping': early_stopping,
        'seed': seed,
        'add_bos_token': add_bos_token,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        # print(prompt + result)
        print(result)
    # return result
