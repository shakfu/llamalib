import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'gemma-2-9b-it-IQ4_XS.gguf'

import pbllama as pb

def ask(prompt, n_predict=512, model=str(MODEL), verbosity=0):
    "ask/prompt a llama model"
    print(pb.simple_prompt(model=model, n_predict=n_predict, prompt=prompt, verbosity=verbosity))


if __name__ == '__main__':
    ask("When did the universe begin?")
