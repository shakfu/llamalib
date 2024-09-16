import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'gemma-2-9b-it-IQ4_XS.gguf'
# MODEL = ROOT / 'models' / 'mistral-7b-instruct-v0.1.Q4_K_M.gguf'

import nbllama as nb

def ask(prompt, model=str(MODEL), n_predict=512, disable_log=True, n_threads=4) -> str:
    """ask/prompt a llama model"""
    return nb.simple_prompt(model=model, n_predict=n_predict, prompt=prompt, disable_log=disable_log,  n_threads=n_threads).strip()


if __name__ == '__main__':
    print(ask("When did the universe begin?"))
