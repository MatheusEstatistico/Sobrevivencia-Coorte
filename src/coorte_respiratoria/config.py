"""
config.py
=========

Configuração central e *pré-registro computacional* do estudo.

Racional metodológico 
---------------------
Em epidemiologia, decisões analíticas tomadas **depois** de ver os resultados
(garden of forking paths) são a principal fonte de inflação do erro tipo I.
Concentrar todos os parâmetros de desenho — janela temporal, critérios de
elegibilidade, ponto de truncamento do seguimento (tau), covariáveis do modelo
ajustado e nível de significância — em um único objeto imutável, versionado no
Git, funciona como um plano de análise estatística (SAP) executável: qualquer
alteração de desenho aparece no `git diff`.

Uso
---
>>> from coorte_respiratoria.config import AnalysisConfig
>>> cfg = AnalysisConfig()                       # padrões do estudo principal
>>> cfg = AnalysisConfig.from_yaml("config/analysis_config.yaml")
>>> cfg.tau_dias
30
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Raiz do projeto: resolvida a partir da localização deste arquivo, de modo que
# o pipeline funcione independentemente do diretório de onde é chamado.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AnalysisConfig:
    """Parâmetros de desenho e análise da coorte.

    `frozen=True` torna a configuração imutável em tempo de execução: nenhuma
    função do pipeline pode alterar silenciosamente um critério de elegibilidade
    no meio da análise.
    """

    # ---------------------------- Desenho ---------------------------------- #
    uf: str = "MG"                      # Unidade federativa (Minas Gerais)
    anos: tuple[int, ...] = (2023,)     # Anos de competência da AIH
    meses: tuple[int, ...] = tuple(range(1, 13))
    grupo_sih: str = "RD"               # RD = AIH Reduzida (o arquivo analítico do SIH)

    # Capítulo X da CID-10 (J00-J99): doenças do aparelho respiratório.
    cid_prefixos: tuple[str, ...] = ("J",)

    idade_minima: int = 18              # Coorte de adultos (pediatria tem hazard distinto)
    idade_maxima: int = 110             # Limite superior plausível (controle de qualidade)

    # Truncamento administrativo do seguimento intra-hospitalar, em dias.
    # 30 dias é o padrão em desfechos hospitalares (mortalidade em 30 dias) e
    # estabiliza a cauda das curvas, onde o n sob risco é pequeno e a variância
    # do estimador de Kaplan-Meier explode.
    tau_dias: float = 30.0

    # Menor unidade de tempo representável. Internações com alta/óbito no mesmo
    # dia têm DIAS_PERM = 0; tempo zero é inadmissível em modelos de sobrevida
    # (log(0)), então imputamos meio dia — equivalente a assumir que o evento
    # ocorreu no ponto médio do intervalo observado.
    tempo_minimo: float = 0.5

    # ------------------------- Estrutura do modelo -------------------------- #
    exposicao_principal: str = "grupo_diagnostico"
    referencia_exposicao: str = "Pneumonia"

    covariaveis_ajuste: tuple[str, ...] = (
        "idade_ns1", "idade_ns2", "idade_ns3",  # spline natural cúbica (3 gl)
        "sexo_feminino",
        "raca_nao_branca",
        "uti",
        "urgencia",
        "comorbidade_registrada",
        "inverno",
    )
    cluster_col: str = "CNES"           # Sanduíche robusto por hospital
    id_col: str = "id_internacao"

    # --------------------------- Inferência --------------------------------- #
    alpha: float = 0.05
    metodo_ajuste_multiplo: str = "holm"   # Holm-Bonferroni (controla FWER)
    n_bootstrap: int = 200                 # Validação interna (otimismo do C-index)
    seed: int = 20240501

    # ----------------------------- Caminhos --------------------------------- #
    dir_raiz: Path = field(default=PROJECT_ROOT)
    dir_dados: Path = field(default=PROJECT_ROOT / "data")
    dir_saidas: Path = field(default=PROJECT_ROOT / "outputs")

    # -------------------- Modo de execução / reprodutibilidade -------------- #
    # 'auto'  -> tenta PySUS; se falhar (sem rede, FTP fora do ar), simula
    # 'pysus' -> exige download real; falha explicitamente se não conseguir
    # 'simular' -> sempre usa o gerador sintético (CI, testes, demonstração)
    modo_dados: str = "auto"
    n_simulacao: int = 40_000

    # ------------------------------------------------------------------ #
    @property
    def dir_figuras(self) -> Path:
        return self.dir_saidas / "figuras"

    @property
    def dir_tabelas(self) -> Path:
        return self.dir_saidas / "tabelas"

    @property
    def dir_modelos(self) -> Path:
        return self.dir_saidas / "modelos"

    def preparar_diretorios(self) -> None:
        """Cria a árvore de saídas (idempotente)."""
        for d in (self.dir_dados / "raw", self.dir_figuras,
                  self.dir_tabelas, self.dir_modelos):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, caminho: str | Path) -> "AnalysisConfig":
        """Carrega a configuração de um YAML, convertendo listas em tuplas.

        Tuplas são usadas porque dataclasses congeladas exigem campos
        *hasheáveis* — o que também impede mutação acidental de listas
        compartilhadas entre módulos.
        """
        with open(caminho, "r", encoding="utf-8") as fh:
            bruto: dict[str, Any] = yaml.safe_load(fh) or {}

        campos_validos = {f for f in cls.__dataclass_fields__}
        limpo: dict[str, Any] = {}
        for k, v in bruto.items():
            if k not in campos_validos:
                raise KeyError(f"Parâmetro desconhecido no YAML: '{k}'")
            if isinstance(v, list):
                v = tuple(v)
            if k.startswith("dir_"):
                v = Path(v)
            limpo[k] = v
        return cls(**limpo)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a configuração (vai para o cabeçalho dos relatórios)."""
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
