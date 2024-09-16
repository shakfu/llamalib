import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'gemma-2-9b-it-IQ4_XS.gguf'
# MODEL = ROOT / 'models' / 'mistral-7b-instruct-v0.1.Q4_K_M.gguf'

from cyllama import ask


if __name__ == '__main__':
    print(ask("When did the universe begin?", model=str(MODEL)))
