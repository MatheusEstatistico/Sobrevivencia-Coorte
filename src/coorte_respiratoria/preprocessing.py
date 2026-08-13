"""
preprocessing.py
================

Do registro administrativo à **coorte analítica**: decodificação do SIH/SUS,
aplicação dos critérios de elegibilidade (com fluxograma auditável) e derivação
das variáveis de tempo e desfecho.

Estrutura de dados do desfecho
------------------------------
A internação por causa respiratória termina em um de três estados mutuamente
exclusivos, o que caracteriza um problema de **riscos competitivos**:

======  ===================================  ============================
Código  Estado                               Fonte no SIH
======  ===================================  ============================
1       Óbito intra-hospitalar               MORTE = 1 ou COBRANCA 41-43
2       Alta hospitalar (vivo)               COBRANCA 11-19
0       Censura                              COBRANCA 31-33 (transferência)
                                             ou permanência além de tau
======  ===================================  ============================

Tratar a alta como "censura" em um Kaplan-Meier de óbito é **incorreto**: a
censura independente exige que o indivíduo censurado permaneça sob risco do
evento, e um paciente que recebeu alta não pode mais morrer *no hospital*. Essa
violação é demonstrada empiricamente em `competing_risks.py`.

Advertência sobre a codificação
-------------------------------
Os mapeamentos abaixo seguem o dicionário de dados do SIH/SUS (DATASUS,
"Estrutura do arquivo de AIH Reduzida - RD"). Layouts variam entre competências
antigas; para séries anteriores a 2008 confira o dicionário da competência.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Dicionários de decodificação do SIH/SUS
# --------------------------------------------------------------------------- #
MAPA_SEXO = {"1": "Masculino", "M": "Masculino", "0": None,
             "3": "Feminino", "2": "Feminino", "F": "Feminino", "9": None}

MAPA_RACA = {"01": "Branca", "02": "Preta", "03": "Parda",
             "04": "Amarela", "05": "Indígena", "99": None, "": None}

MAPA_CAR_INT = {"01": "Eletivo", "02": "Urgência", "03": "Acidente trabalho",
                "04": "Acidente trajeto", "05": "Acidente trânsito",
                "06": "Lesão/envenenamento"}

# COD_IDADE: unidade em que o campo IDADE está expresso.
FATOR_IDADE = {"0": 1 / (365.25 * 24), "1": 1 / (365.25 * 24),  # horas/minutos
               "2": 1 / 365.25,        # dias
               "3": 1 / 12,            # meses
               "4": 1.0, "5": 1.0}     # anos (5 = 100 anos ou mais)

# --------------------------------------------------------------------------- #
# Agrupamento clínico da CID-10, capítulo X (J00-J99)
# Os grupos foram definidos a priori por homogeneidade fisiopatológica e
# prognóstica, seguindo a lógica das "Ambulatory Care Sensitive Conditions"
# e da lista brasileira de internações por condições sensíveis à APS.
# --------------------------------------------------------------------------- #
def _num_cid(codigo: str) -> float:
    """Extrai a parte numérica de um código CID-10 ('J189' -> 18.9)."""
    m = re.match(r"^J(\d{2})(\d?)$", codigo)
    if not m:
        return np.nan
    return float(m.group(1)) + (float(m.group(2)) / 10 if m.group(2) else 0.0)


def agrupar_cid_respiratorio(codigo: str) -> str | float:
    """Mapeia um código J** para o grupo diagnóstico do estudo."""
    n = _num_cid(codigo)
    if np.isnan(n):
        return np.nan
    if 12 <= n < 19:
        return "Pneumonia"                    # J12-J18
    if 40 <= n < 45:
        return "DPOC"                         # J40-J44 (bronquite crônica, enfisema, DPOC)
    if 45 <= n < 47:
        return "Asma"                         # J45-J46
    if (80 <= n < 85) or (96 <= n < 97):
        return "Insuf. respiratória"          # J80-J84 (SDRA, intersticiais), J96
    return "Outras"                           # demais J (IVAS, pleurais, ocupacionais...)


GRUPOS_DIAG = ["Pneumonia", "DPOC", "Asma", "Insuf. respiratória", "Outras"]


# =========================================================================== #
# Utilitários
# =========================================================================== #
def _para_texto(s: pd.Series, largura: int | None = None) -> pd.Series:
    """Normaliza campos categóricos do DATASUS: string, sem espaços, maiúscula.

    O DATASUS entrega códigos ora como inteiro, ora como string zero-padded,
    e frequentemente com espaços à direita (herança do formato DBF de largura
    fixa). Sem essa normalização, `RACA_COR == '01'` falha silenciosamente.
    """
    out = s.astype("string").str.strip().str.upper()
    if largura:
        out = out.str.zfill(largura)
    return out


def _para_data(s: pd.Series) -> pd.Series:
    """Converte AAAAMMDD (string ou int) em datetime, com coerção de inválidos."""
    return pd.to_datetime(_para_texto(s), format="%Y%m%d", errors="coerce")


def _para_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_para_texto(s), errors="coerce")


# =========================================================================== #
# Pipeline principal
# =========================================================================== #
def construir_coorte(df_bruto: pd.DataFrame,
                     cfg: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transforma a AIH-RD bruta na coorte analítica.

    Returns
    -------
    (df, fluxo)
        `df` é a coorte analítica; `fluxo` é o dicionário do fluxograma de
        elegibilidade (equivalente à Figura 1 de um artigo STROBE), que
        registra quantas AIH foram excluídas em cada etapa e por quê.
    """
    df = df_bruto.copy()
    fluxo: dict[str, int] = {"AIH recuperadas do SIH/SUS": len(df)}

    # ------------------------------------------------------------------ #
    # 1. Normalização de tipos
    # ------------------------------------------------------------------ #
    for col, larg in [("SEXO", None), ("RACA_COR", 2), ("CAR_INT", 2),
                      ("MARCA_UTI", 2), ("COBRANCA", 2), ("IDENT", None),
                      ("COMPLEX", 2), ("ESPEC", 2), ("CNES", 7)]:
        if col in df:
            df[col] = _para_texto(df[col], larg)

    for col in ("DIAG_PRINC", "DIAG_SECUN", "MUNIC_RES", "N_AIH"):
        if col in df:
            # Códigos CID no DATASUS podem vir com espaço interno ('J38 3').
            df[col] = _para_texto(df[col]).str.replace(r"[^A-Z0-9]", "", regex=True)

    for col in ("DT_INTER", "DT_SAIDA", "NASC"):
        if col in df:
            df[col] = _para_data(df[col])

    for col in ("DIAS_PERM", "IDADE", "MORTE", "UTI_MES_TO", "VAL_TOT"):
        if col in df:
            df[col] = _para_num(df[col])

    # ------------------------------------------------------------------ #
    # 2. Elegibilidade — cada filtro é registrado no fluxograma
    # ------------------------------------------------------------------ #
    # 2.1 AIH de continuação (IDENT = 5) representam prorrogação de uma
    # internação já contabilizada; incluí-las duplicaria pacientes e violaria
    # a independência das observações.
    if "IDENT" in df:
        df = df[df["IDENT"].isin(["1", "01", "1.0", ""]) | df["IDENT"].isna()]
        fluxo["Após excluir AIH de continuação (IDENT=5)"] = len(df)

    # 2.2 Diagnóstico principal respiratório (capítulo X da CID-10)
    padrao = tuple(cfg.cid_prefixos)
    df = df[df["DIAG_PRINC"].fillna("").str.startswith(padrao)]
    fluxo[f"Diagnóstico principal {'/'.join(padrao)}00-99"] = len(df)

    # 2.3 Idade em anos completos
    if "COD_IDADE" in df:
        fator = _para_texto(df["COD_IDADE"]).map(FATOR_IDADE).fillna(1.0)
    else:
        fator = 1.0
    df["idade"] = df["IDADE"] * fator
    # Quando NASC e DT_INTER existem, a idade calculada é mais confiável
    if {"NASC", "DT_INTER"} <= set(df.columns):
        idade_calc = (df["DT_INTER"] - df["NASC"]).dt.days / 365.25
        df["idade"] = df["idade"].where(idade_calc.isna(), idade_calc)

    df = df[df["idade"].between(cfg.idade_minima, cfg.idade_maxima)]
    fluxo[f"Idade {cfg.idade_minima}-{cfg.idade_maxima} anos"] = len(df)

    # 2.4 Tempo de permanência válido
    if {"DT_INTER", "DT_SAIDA"} <= set(df.columns):
        los_data = (df["DT_SAIDA"] - df["DT_INTER"]).dt.days
        df["los"] = np.where(los_data.notna() & (los_data >= 0), los_data, df["DIAS_PERM"])
    else:
        df["los"] = df["DIAS_PERM"]
    df = df[df["los"].notna() & (df["los"] >= 0) & (df["los"] <= 365)]
    fluxo["Permanência válida (0-365 dias)"] = len(df)

    # ------------------------------------------------------------------ #
    # 3. Desfecho competitivo
    # ------------------------------------------------------------------ #
    cob = df["COBRANCA"] if "COBRANCA" in df else pd.Series("", index=df.index)
    cob_num = pd.to_numeric(cob, errors="coerce")

    obito = (df.get("MORTE", 0) == 1) | cob_num.between(41, 43)
    transferencia = cob_num.between(31, 33)
    alta = cob_num.between(11, 19) & ~obito

    # Hierarquia: óbito domina (o registro de óbito é o mais confiável do SIH,
    # pois vincula-se à emissão da Declaração de Óbito); depois transferência;
    # o restante que não é óbito nem transferência é tratado como alta viva.
    df["status"] = np.select(
        [obito, transferencia, alta],
        [1, 0, 2],
        default=2,
    ).astype(int)
    df["motivo_censura"] = np.where(transferencia & ~obito, "Transferência", "")

    # ------------------------------------------------------------------ #
    # 4. Escala de tempo e truncamento administrativo em tau
    # ------------------------------------------------------------------ #
    # Origem do tempo = data de internação (t = 0). Não há truncamento à
    # esquerda porque a coorte é definida no momento da admissão.
    df["tempo"] = df["los"].clip(lower=cfg.tempo_minimo)

    excede = df["tempo"] > cfg.tau_dias
    df.loc[excede, "status"] = 0                      # ainda internado em tau
    df.loc[excede, "motivo_censura"] = "Administrativa (tau)"
    df["tempo"] = df["tempo"].clip(upper=cfg.tau_dias)

    # ------------------------------------------------------------------ #
    # 5. Covariáveis analíticas
    # ------------------------------------------------------------------ #
    df["grupo_diagnostico"] = df["DIAG_PRINC"].map(agrupar_cid_respiratorio)
    df["grupo_diagnostico"] = pd.Categorical(
        df["grupo_diagnostico"], categories=GRUPOS_DIAG, ordered=False)

    df["sexo"] = _para_texto(df["SEXO"]).map(MAPA_SEXO) if "SEXO" in df else None
    df["sexo_feminino"] = (df["sexo"] == "Feminino").astype(int)

    df["raca"] = df["RACA_COR"].map(MAPA_RACA) if "RACA_COR" in df else None
    # Dicotomização branca vs não branca: as categorias amarela/indígena têm
    # n insuficiente para estimativas estáveis em MG; a análise de sensibilidade
    # usa as cinco categorias com imputação múltipla.
    df["raca_nao_branca"] = np.where(df["raca"].isna(), np.nan,
                                     (df["raca"] != "Branca").astype(float))

    df["uti"] = 0
    if "MARCA_UTI" in df:
        df["uti"] = (~df["MARCA_UTI"].isin(["00", "0", ""])).astype(int)
    if "UTI_MES_TO" in df:
        df["uti"] = ((df["uti"] == 1) | (df["UTI_MES_TO"].fillna(0) > 0)).astype(int)
    df["dias_uti"] = df.get("UTI_MES_TO", pd.Series(0, index=df.index)).fillna(0)

    df["carater"] = df["CAR_INT"].map(MAPA_CAR_INT) if "CAR_INT" in df else None
    df["urgencia"] = (df.get("CAR_INT", pd.Series("", index=df.index)) != "01").astype(int)

    # Proxy de comorbidade: presença de diagnóstico secundário válido.
    # Limitação relevante — o SIH registra apenas um CID secundário, o que
    # subestima a carga de comorbidade (ver Discussão).
    ds = df.get("DIAG_SECUN", pd.Series("", index=df.index)).fillna("")
    df["comorbidade_registrada"] = (ds.str.len().ge(3) & ~ds.str.match(r"^0+$")).astype(int)

    df["mes_internacao"] = df["DT_INTER"].dt.month if "DT_INTER" in df else np.nan
    # No hemisfério sul, o pico de circulação de vírus respiratórios em MG
    # ocorre entre maio e agosto (SRAG - InfoGripe/Fiocruz).
    df["inverno"] = df["mes_internacao"].isin([5, 6, 7, 8]).astype(int)

    df["alta_complexidade"] = (df.get("COMPLEX", pd.Series("", index=df.index)) == "03").astype(int)
    df["id_internacao"] = np.arange(len(df))
    if "CNES" not in df:
        df["CNES"] = "unico"

    # Spline natural cúbica para idade (3 gl) — ver survival_cox.build_splines
    from .survival_cox import base_spline_natural
    base = base_spline_natural(df["idade"].to_numpy(float), gl=3)
    for j in range(base.shape[1]):
        df[f"idade_ns{j+1}"] = base[:, j]

    # ------------------------------------------------------------------ #
    # 6. Casos completos para o modelo ajustado
    # ------------------------------------------------------------------ #
    fluxo["Coorte analítica"] = len(df)
    n_incompleto = df[["raca_nao_branca", "sexo_feminino"]].isna().any(axis=1).sum()
    fluxo["  (com raça/cor ausente — mantidos, imputados na sensibilidade)"] = int(n_incompleto)

    df = df.reset_index(drop=True)
    df.attrs["fluxo"] = fluxo
    df.attrs["origem"] = df_bruto.attrs.get("origem", "desconhecida")

    log.info("Coorte construída: %d internações | %d óbitos | %d altas | %d censuras",
             len(df), (df.status == 1).sum(), (df.status == 2).sum(), (df.status == 0).sum())
    return df, fluxo


def formatar_fluxograma(fluxo: dict[str, int]) -> str:
    """Renderiza o fluxograma de elegibilidade em texto (Figura 1 do artigo)."""
    def milhar(n: int) -> str:
        # Separador de milhar no padrão pt-BR. Formatar apenas o número: um
        # `str.replace(",", ".")` aplicado à linha inteira corromperia a
        # pontuação do rótulo (bug real encontrado em teste).
        return f"{n:,}".replace(",", ".")

    linhas, anterior = ["FLUXOGRAMA DE ELEGIBILIDADE (STROBE, Figura 1)", "=" * 62], None
    for etapa, n in fluxo.items():
        if anterior is None or etapa.startswith("  "):
            linhas.append(f"{etapa:.<52} {milhar(n):>8}")
        else:
            linhas.append(f"{etapa:.<52} {milhar(n):>8}   (−{milhar(anterior - n)})")
        if not etapa.startswith("  "):
            anterior = n
    return "\n".join(linhas)
