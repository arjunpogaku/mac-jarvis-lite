test:
	python -m pytest -q

self-check:
	python scripts/self_check.py

smoke:
	python scripts/smoke_test.py

quality:
	python -m pytest -q
	python scripts/self_check.py
	python scripts/smoke_test.py
