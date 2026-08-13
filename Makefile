.PHONY: help setup test lint run run-real clean

help:
	@echo "setup     Instala as dependências"
	@echo "test      Roda a suíte de testes"
	@echo "run       Executa o pipeline (simula se não houver rede)"
	@echo "run-real  Executa exigindo download real do DATASUS via PySUS"
	@echo "quick     Execução rápida (menos reamostragens) — usada na CI"
	@echo "clean     Remove saídas geradas"

setup:
	python -m pip install -U pip
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

quick:
	python run_analysis.py --modo simular --n-sim 8000 --rapido

run:
	python run_analysis.py --config config/analysis_config.yaml

run-real:
	python run_analysis.py --config config/analysis_config.yaml --modo pysus

clean:
	rm -rf outputs/figuras/* outputs/tabelas/* outputs/*.parquet \
	       outputs/*.json outputs/RESULTADOS.md
