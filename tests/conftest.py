import sys
from pathlib import Path
ROOT = Path.cwd()


import pytest

@pytest.fixture(scope="module")
def MODEL():
	return str(ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf')
