import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'

import pbllama as pb

def ask(prompt, model=str(MODEL), n_predict=512, disable_log=True, n_threads=4) -> str:
    """ask/prompt a llama model"""
    return pb.simple_prompt(model=model, n_predict=n_predict, prompt=prompt, disable_log=disable_log,  n_threads=n_threads).strip()


if __name__ == '__main__':
    print(ask("When did the universe begin?"))
