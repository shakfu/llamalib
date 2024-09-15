import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'build'))

import scratch

def test_scratch():

	p = scratch.WrappedPerson(10, [1.0, 2.1])

	print(p.grades)

	assert True