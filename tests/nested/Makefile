
.PHONY: all build clean

all: build clean reset

build:
	@python3 setup.py build_ext --inplace

clean:
	@rm -rf build demo.c

reset: clean
	@rm -f *.so
