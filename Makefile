.PHONY: test experiments paper all clean

test:
	pytest

experiments:
	PYTHONPATH=src python experiments/run_all.py

paper:
	cd paper && ./build.sh

all: test experiments paper

clean:
	rm -rf .pytest_cache src/*.egg-info src/mnar_rl/__pycache__ tests/__pycache__
	cd paper && latexmk -C main.tex
