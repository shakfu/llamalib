import sys
from pathlib import Path
ROOT = Path.cwd()


import pytest

@pytest.fixture(scope="module")
def MODEL():
	return str(ROOT / 'models' / 'gemma-2-9b-it-IQ4_XS.gguf')
