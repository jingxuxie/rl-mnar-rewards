.PHONY: test experiments paper submission-check clean

test:
	pytest

experiments:
	python experiments/run_all.py --random-mdps 200 --replicates 1000 --post-selection-replicates 3000 --gamma-mdps 100

paper:
	cd paper && bash build.sh

submission-check:
	cd paper && AAAI_MODE=review bash build.sh
	python scripts/check_submission.py

clean:
	rm -rf .pytest_cache src/mnar_rl/__pycache__ tests/__pycache__ experiments/__pycache__
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.pdf
