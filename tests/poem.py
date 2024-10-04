"""poetry generator

"""

import os
import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

from tqdm import tqdm

pre_prompt = "Provide a Haiku of three lines with a syllable count of 5-7-5 about "
subject = "the taste of the first cup of coffee in the morning of a wintry day."
prompt = pre_prompt + subject

models = [
    'Cinder-Phi-2-V1.F16.gguf',
    'DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q6_K.gguf',
    'Gemma-2-Ataraxy-v2-9B-Q5_K_M.gguf',
    'Llama-3.2-1B-Instruct-Q6_K.gguf',
    'Llama-3.2-1B-Instruct-Q8_0.gguf',
    'Llama-3.2-3B-Instruct-Q6_K.gguf',
    'Llama-3.2-3B-Instruct-Q8_0.gguf',
    'Meta-Llama-3.1-8B-Instruct-Q6_K.gguf',
    'OLMo-7B-Instruct-Q6_K.gguf',
    'Phi-3.5-mini-instruct.Q6_K.gguf',
    'Phi-3.5-mini-instruct_Uncensored-Q6_K_L.gguf',
    'gemma-2-9b-it-IQ4_XS.gguf',
    'llama-3.2-1b-instruct-q4_k_m.gguf',
    'mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    'open-hermes-sd-finetune-erot-story.i1-Q5_K_S.gguf',
    'vicuna-7b-cot.Q5_K_M.gguf',
]

from cyllama import ask

results = {}

for model in tqdm(models):
    model_path = str(ROOT / 'models' / model)
    results[model] = ask(prompt, model_path)
    # print(f"done: {model}")


for model, poem in results.items():
    print(f"## {model}")
    print()
    print(poem)
    print()
    print()
