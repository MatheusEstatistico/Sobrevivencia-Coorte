"""
descriptive.py
==============

Tabela 1 (características basais) e diagnóstico de dados ausentes.

Por que diferenças padronizadas e não p-valores?
------------------------------------------------
Comparar grupos basais com testes de hipótese é uma prática desencorajada em
estudos observacionais (Austin, 2009; Senn, 1994): com n grande — e aqui há
dezenas de milhares de AIH — diferenças clinicamente irrelevantes atingem
p < 0,001, enquanto em subgrupos pequenos desequilíbrios importantes passam
"não significativos". O p-valor mede evidência contra a hipótese de
equivalência *na população*, mas a coorte **é** a população de interesse; a
pergunta relevante é sobre o tamanho do desequilíbrio na amostra observada.

A **diferença média padronizada (SMD)** responde a isso e independe do n:

.. math::
   \\mathrm{SMD} = \\frac{|\\bar x_1 - \\bar x_2|}
                        {\\sqrt{(s_1^2 + s_2^2)/2}}
   \\qquad
   \\mathrm{SMD}_{\\text{binária}} = \\frac{|p_1 - p_2|}
        {\\sqrt{[p_1(1-p_1) + p_2(1-p_2)]/2}}

Convenção de leitura: SMD > 0,10 indica desequilíbrio não desprezível.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def smd_continua(x1: np.ndarray, x2: np.ndarray) -> float:
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    s = np.sqrt((np.var(x1, ddof=1) + np.var(x2, ddof=1)) / 2)
    return float(abs(x1.mean() - x2.mean()) / s) if s > 0 else 0.0


def smd_binaria(p1: float, p2: float) -> float:
    d = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return float(abs(p1 - p2) / d) if d > 0 else 0.0


def _fmt(v: float, casas: int = 1) -> str:
    return f"{v:,.{casas}f}".replace(",", "@").replace(".", ",").replace("@", ".")


# --------------------------------------------------------------------------- #
def tabela_um(df: pd.DataFrame,
              grupo: str,
              continuas: Sequence[str],
              categoricas: Sequence[str],
              referencia: str | None = None,
              rotulos: dict[str, str] | None = None) -> pd.DataFrame:
    """Constrói a Tabela 1 com coluna Total, colunas por grupo e SMD máxima.

    Variáveis contínuas são resumidas por **mediana [IQR]** quando assimétricas
    (assimetria |g1| > 1) e por média (DP) caso contrário — a escolha é feita
    por variável e registrada na própria linha, evitando o erro comum de
    reportar média ± DP para tempo de permanência, que é fortemente
    assimétrico à direita.
    """
    rotulos = rotulos or {}
    grupos = [g for g in df[grupo].dropna().unique()]
    if referencia and referencia in grupos:
        grupos = [referencia] + [g for g in grupos if g != referencia]
    partes = {str(g): df[df[grupo] == g] for g in grupos}

    linhas: list[dict] = []
    linhas.append({"Característica": "n", "Total": f"{len(df):,}".replace(",", "."),
                   **{k: f"{len(v):,}".replace(",", ".") for k, v in partes.items()},
                   "SMD máx.": ""})

    for v in continuas:
        if v not in df:
            continue
        x = pd.to_numeric(df[v], errors="coerce")
        assimetrica = abs(x.skew()) > 1
        nome = rotulos.get(v, v) + (" — mediana [IQR]" if assimetrica else " — média (DP)")

        def resumo(s: pd.Series) -> str:
            s = pd.to_numeric(s, errors="coerce").dropna()
            if not len(s):
                return "—"
            if assimetrica:
                return f"{_fmt(s.median())} [{_fmt(s.quantile(.25))}–{_fmt(s.quantile(.75))}]"
            return f"{_fmt(s.mean())} ({_fmt(s.std())})"

        smds = [smd_continua(pd.to_numeric(a[v], errors="coerce").to_numpy(float),
                             pd.to_numeric(b[v], errors="coerce").to_numpy(float))
                for i, a in enumerate(partes.values())
                for j, b in enumerate(partes.values()) if j > i]
        linhas.append({"Característica": nome, "Total": resumo(x),
                       **{k: resumo(d[v]) for k, d in partes.items()},
                       "SMD máx.": _fmt(np.nanmax(smds), 2) if smds else ""})

    for v in categoricas:
        if v not in df:
            continue
        linhas.append({"Característica": rotulos.get(v, v) + " — n (%)", "Total": "",
                       **{k: "" for k in partes}, "SMD máx.": ""})
        niveis = pd.Series(df[v].dropna().unique()).astype(str).sort_values()
        for nivel in niveis:
            def prop(d: pd.DataFrame) -> tuple[int, float]:
                s = d[v].astype(str)
                n = int((s == nivel).sum())
                return n, n / max(len(d.dropna(subset=[v])), 1)

            n_tot, p_tot = prop(df)
            props = {k: prop(d) for k, d in partes.items()}
            smds = [smd_binaria(a[1], b[1])
                    for i, a in enumerate(props.values())
                    for j, b in enumerate(props.values()) if j > i]
            linhas.append({
                "Característica": f"   {nivel}",
                "Total": f"{n_tot:,}".replace(",", ".") + f" ({_fmt(100 * p_tot)}%)",
                **{k: f"{n:,}".replace(",", ".") + f" ({_fmt(100 * p)}%)"
                   for k, (n, p) in props.items()},
                "SMD máx.": _fmt(np.nanmax(smds), 2) if smds else ""})

    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
def relatorio_ausencia(df: pd.DataFrame, variaveis: Sequence[str]) -> pd.DataFrame:
    """Quantifica dados ausentes e testa se a ausência é associada ao desfecho.

    Se a probabilidade de ausência difere entre quem morreu e quem não morreu,
    o mecanismo não é MCAR e a análise de casos completos pode ser enviesada —
    justificando a imputação múltipla na análise de sensibilidade.
    """
    linhas = []
    obito = (df["status"] == 1).astype(int)
    for v in variaveis:
        if v not in df:
            continue
        aus = df[v].isna()
        if aus.sum() == 0:
            linhas.append({"Variável": v, "Ausentes": 0, "% ausente": 0.0,
                           "% ausente entre óbitos": 0.0,
                           "% ausente entre não óbitos": 0.0, "Mecanismo sugerido": "Completo"})
            continue
        p1 = float(aus[obito == 1].mean()) if (obito == 1).any() else np.nan
        p0 = float(aus[obito == 0].mean()) if (obito == 0).any() else np.nan
        mecanismo = "Possível MAR/MNAR" if abs(p1 - p0) > 0.02 else "Compatível com MCAR"
        linhas.append({"Variável": v, "Ausentes": int(aus.sum()),
                       "% ausente": 100 * float(aus.mean()),
                       "% ausente entre óbitos": 100 * p1,
                       "% ausente entre não óbitos": 100 * p0,
                       "Mecanismo sugerido": mecanismo})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
def tabela_incidencia(df: pd.DataFrame, grupo: str, tau: float) -> pd.DataFrame:
    """Densidade de incidência de óbito por 1.000 pessoas-dia de internação.

    Complementa as medidas de risco: enquanto a CIF responde "qual a chance de
    morrer em 30 dias", a taxa responde "com que intensidade os óbitos ocorrem
    por unidade de tempo sob risco" — útil quando a permanência difere muito
    entre grupos, como aqui (asma tem alta rápida, DPOC não).
    """
    linhas = []
    for nivel, g in df.groupby(grupo, observed=True):
        pd_dias = float(g["tempo"].sum())
        ob = int((g["status"] == 1).sum())
        taxa = 1000 * ob / pd_dias if pd_dias > 0 else np.nan
        # IC de Poisson (Byar) para a taxa
        ic_inf = 1000 * (ob * (1 - 1 / (9 * ob) - 1.96 / (3 * np.sqrt(ob))) ** 3) / pd_dias if ob > 0 else 0
        ic_sup = 1000 * ((ob + 1) * (1 - 1 / (9 * (ob + 1)) + 1.96 / (3 * np.sqrt(ob + 1))) ** 3) / pd_dias
        linhas.append({"Grupo": str(nivel), "Internações": len(g), "Óbitos": ob,
                       "Pessoas-dia": pd_dias,
                       "Taxa /1.000 pessoas-dia": taxa,
                       "IC95% inf": ic_inf, "IC95% sup": ic_sup})
    return pd.DataFrame(linhas)
