"""poetry generator

"""

import os
import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

HAVE_TQDM = False
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    tqdm = lambda x: x


pre_prompt = "Provide a Haiku of three lines with a syllable count of 5-7-5 about "
subject = "the first cup of coffee in the morning."
prompt = pre_prompt + subject

models = [
    'Chronos-Gold-12B-1.0.Q4_K_M.gguf',
    'Cinder-Phi-2-V1.F16.gguf',
    'DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q6_K.gguf',
    'Gemma-2-Ataraxy-v2-9B-Q5_K_M.gguf',
    'Llama-3.2-1B-Instruct-Q6_K.gguf',
    'Llama-3.2-1B-Instruct-Q8_0.gguf',
    'Llama-3.2-3B-Instruct-Q6_K.gguf',
    'Llama-3.2-3B-Instruct-Q8_0.gguf',
    'LongWriter-llama3.1-8b-Q5_K_M.gguf',
    'Meta-Llama-3.1-8B-Instruct-Q6_K.gguf',
    'OLMo-7B-Instruct-Q6_K.gguf',
    'Phi-3.5-mini-instruct.Q6_K.gguf',
    'Phi-3.5-mini-instruct_Uncensored-Q6_K_L.gguf',
    'SmolLM-1.7B-Instruct-v0.2.Q8_0.gguf',
    'gemma-2-9b-it-IQ4_XS.gguf',
    'llama-3.2-1b-instruct-q4_k_m.gguf',
    'mamba-gpt-3B_v4.gguf',
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

with open('haiku.md', 'w') as f:
    for model, poem in results.items():
        f.write(f"## {model}\n\n")
        f.write(f"{poem}\n\n\n")
