
.PHONY: all build clean reset

all: build clean

build:
	@python3 setup.py build_ext --inplace

clean:
	@rm -rf build demo.cpp

reset: clean
	@rm -f *.so
