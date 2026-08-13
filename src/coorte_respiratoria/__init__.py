"""
coorte_respiratoria
===================

Pipeline reprodutível de **análise de sobrevida em coorte clínica** construída a
partir do Sistema de Informações Hospitalares do SUS (SIH/SUS), com foco em
internações por doenças do aparelho respiratório (CID-10 J00-J99) em Minas
Gerais.

Módulos
-------
``config``            Parâmetros de desenho (plano de análise executável).
``data_acquisition``  Download via PySUS, cache e simulador de contingência.
``preprocessing``     Decodificação do SIH, elegibilidade, desfecho competitivo.
``descriptive``       Tabela 1, SMD, dados ausentes, densidade de incidência.
``survival_km``       Kaplan-Meier, log-rank, RMST.
``survival_cox``      Cox com splines, Schoenfeld, SE robusto, bootstrap.
``competing_risks``   Aalen-Johansen, causa específica e Fine-Gray.
``viz``               Figuras em padrão de publicação.

Uso mínimo
----------
>>> from coorte_respiratoria import AnalysisConfig, obter_coorte, construir_coorte
>>> cfg = AnalysisConfig()
>>> bruto = obter_coorte(cfg)
>>> coorte, fluxo = construir_coorte(bruto, cfg)
"""

from .config import AnalysisConfig, PROJECT_ROOT
from .data_acquisition import obter_coorte, simular_sih
from .preprocessing import construir_coorte, formatar_fluxograma

__all__ = ["AnalysisConfig", "PROJECT_ROOT", "obter_coorte", "simular_sih",
           "construir_coorte", "formatar_fluxograma"]
__version__ = "1.0.0" # Upada para o github
