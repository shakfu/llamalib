import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'

from cyllama import ask


if __name__ == '__main__':
    print(ask("When did the universe begin?", model=str(MODEL)))
