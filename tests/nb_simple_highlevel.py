import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'gemma-2-9b-it-IQ4_XS.gguf'
# MODEL = ROOT / 'models' / 'mistral-7b-instruct-v0.1.Q4_K_M.gguf'

import nbllama as nb

def ask(prompt, n_predict=512, model=str(MODEL), disable_log=True):
    "ask/prompt a llama model"
    print(nb.simple_prompt(model=model, n_predict=n_predict, prompt=prompt, disable_log=disable_log))


if __name__ == '__main__':
    ask("When did the universe begin?")
