"""
survival_km.py
==============

Estimação não paramétrica: Kaplan-Meier estratificado, testes de log-rank
(global e par a par com controle de multiplicidade) e **tempo médio restrito
de sobrevida (RMST)**.

Por que também RMST?
--------------------
A razão de riscos (HR) é o resumo mais reportado, mas só tem interpretação
causal simples se os riscos forem proporcionais. Quando não são — situação
frequente em desfechos hospitalares, onde o efeito de um diagnóstico se
concentra nos primeiros dias — o HR estimado vira uma média ponderada dos HR
instantâneos, com pesos que dependem da distribuição de censura da própria
amostra (Hernán, 2010).

O RMST não sofre disso. Ele é literalmente a área sob a curva de sobrevida até
um horizonte :math:`\\tau`:

.. math::
    \\mathrm{RMST}(\\tau) = \\int_0^{\\tau} S(u)\\,du

e se interpreta como "número médio de dias vividos, no hospital, dentro dos
primeiros :math:`\\tau` dias". A diferença de RMST entre grupos é uma medida de
efeito **em unidade de tempo**, válida mesmo sob não proporcionalidade, e é
recomendada pelo FDA/EMA como análise de suporte.

Advertência importante
----------------------
As curvas de "sobrevida ao óbito" produzidas aqui tratam a alta hospitalar como
censura e, portanto, **superestimam o risco de óbito** na presença do evento
competitivo. Elas são apresentadas com finalidade didática e de comparação —
a estimativa válida de risco absoluto é a incidência acumulada de
Aalen-Johansen (`competing_risks.py`). O KM da variável "permanência
hospitalar" (evento = qualquer saída) permanece válido, pois aí não há
competição.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from statsmodels.stats.multitest import multipletests

from .config import AnalysisConfig

log = logging.getLogger(__name__)


# =========================================================================== #
# 1. Curvas estratificadas
# =========================================================================== #
@dataclass
class ResultadoKM:
    curvas: dict[str, KaplanMeierFitter]
    tabela_risco: pd.DataFrame          # n sob risco por tempo (rodapé da figura)
    logrank_global: pd.DataFrame
    logrank_pareado: pd.DataFrame
    rmst: pd.DataFrame
    rmst_contrastes: pd.DataFrame
    resumo: pd.DataFrame


def ajustar_km(df: pd.DataFrame, cfg: AnalysisConfig,
               estrato: str | None = None,
               evento_alvo: int = 1) -> dict[str, KaplanMeierFitter]:
    """Ajusta uma curva de Kaplan-Meier por nível do estrato.

    Parameters
    ----------
    evento_alvo : int
        1 = óbito intra-hospitalar (alta tratada como censura — ver advertência
        no cabeçalho do módulo);
        2 = alta hospitalar;
        -1 = qualquer saída do hospital (óbito **ou** alta): é a análise de
        *tempo de permanência*, sem competição e portanto sem viés.
    """
    estrato = estrato or cfg.exposicao_principal
    curvas: dict[str, KaplanMeierFitter] = {}

    for nivel, g in df.groupby(estrato, observed=True):
        if len(g) < 20:
            log.warning("Estrato '%s' com n=%d: ignorado (instável).", nivel, len(g))
            continue
        evento = (g["status"] > 0).astype(int) if evento_alvo == -1 \
            else (g["status"] == evento_alvo).astype(int)
        km = KaplanMeierFitter(label=str(nivel))
        km.fit(g["tempo"], evento)
        curvas[str(nivel)] = km
    return curvas


def tabela_numeros_em_risco(df: pd.DataFrame, estrato: str,
                            tempos: Sequence[float]) -> pd.DataFrame:
    """Tabela "number at risk" — item obrigatório em figuras de sobrevida
    segundo as diretrizes de reporte (e a principal defesa do leitor contra
    interpretar caudas construídas com 3 pacientes)."""
    linhas = {}
    for nivel, g in df.groupby(estrato, observed=True):
        linhas[str(nivel)] = [int((g["tempo"] >= t).sum()) for t in tempos]
    return pd.DataFrame(linhas, index=[f"{t:g}" for t in tempos]).T


# =========================================================================== #
# 2. Testes de log-rank
# =========================================================================== #
def testes_logrank(df: pd.DataFrame, cfg: AnalysisConfig,
                   estrato: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log-rank global (k amostras) + comparações par a par com ajuste de Holm.

    O log-rank é o teste de escore do modelo de Cox sob H0 e tem poder máximo
    quando os riscos são proporcionais. Com k = 5 grupos há 10 comparações
    par a par; sem correção, a probabilidade de ao menos um falso positivo a
    5% seria ≈ 40%. Usamos **Holm-Bonferroni**, que controla o FWER sem
    assumir independência entre os testes (os pares compartilham dados, logo
    são correlacionados — o que inviabiliza Benjamini-Hochberg ingênuo).
    """
    estrato = estrato or cfg.exposicao_principal
    d = df.dropna(subset=[estrato])
    evento = (d["status"] == 1).astype(int)

    g = multivariate_logrank_test(d["tempo"], d[estrato].astype(str), evento)
    global_df = pd.DataFrame({"estatistica_qui2": [g.test_statistic],
                              "gl": [g.degrees_of_freedom],
                              "p": [g.p_value]})

    pw = pairwise_logrank_test(d["tempo"], d[estrato].astype(str), evento)
    tab = pw.summary.reset_index()
    if len(tab):
        rej, p_aj, _, _ = multipletests(tab["p"].to_numpy(),
                                        alpha=cfg.alpha,
                                        method=cfg.metodo_ajuste_multiplo)
        tab["p_ajustado"] = p_aj
        tab["significativo"] = rej
    return global_df, tab


# =========================================================================== #
# 3. RMST e sua variância
# =========================================================================== #
def rmst_km(tempo: np.ndarray, evento: np.ndarray, tau: float) -> tuple[float, float]:
    """RMST e sua variância assintótica a partir do estimador de Kaplan-Meier.

    Implementação direta (Klein & Moeschberger, 2003, §4.5):

    .. math::
        \\widehat{\\mathrm{RMST}} = \\sum_j \\hat S(t_{j-1})\\,(t_j - t_{j-1})

    .. math::
        \\widehat{\\mathrm{Var}} = \\sum_{t_i \\le \\tau}
            \\left[\\int_{t_i}^{\\tau} \\hat S(u)du\\right]^2
            \\frac{d_i}{n_i (n_i - d_i)}

    A variância usa a "área residual" a partir de cada tempo de evento, o que
    dá o análogo de Greenwood para a área sob a curva.
    """
    tempo = np.asarray(tempo, float)
    evento = np.asarray(evento, int)
    ordem = np.argsort(tempo)
    tempo, evento = tempo[ordem], evento[ordem]

    t_unicos = np.unique(tempo[(evento == 1) & (tempo <= tau)])
    n_total = len(tempo)

    S, s_atual = [], 1.0
    n_risco, d_evt = [], []
    for t in t_unicos:
        n_i = int((tempo >= t).sum())
        d_i = int(((tempo == t) & (evento == 1)).sum())
        n_risco.append(n_i)
        d_evt.append(d_i)
        s_atual *= (1 - d_i / n_i) if n_i > 0 else 1.0
        S.append(s_atual)

    if not t_unicos.size:
        return float(tau), 0.0

    # Área sob a curva escada: S é constante em [t_j, t_{j+1})
    bordas = np.concatenate(([0.0], t_unicos, [tau]))
    niveis = np.concatenate(([1.0], np.array(S)))
    larguras = np.diff(bordas)
    area = float(np.sum(niveis * larguras))

    # Variância
    var = 0.0
    for i, t in enumerate(t_unicos):
        n_i, d_i = n_risco[i], d_evt[i]
        if n_i - d_i <= 0:
            continue
        # área de t_i até tau
        b = np.concatenate(([t], t_unicos[t_unicos > t], [tau]))
        lv = np.concatenate(([niveis[i + 1]], np.array(S)[t_unicos > t]))
        area_residual = float(np.sum(lv * np.diff(b)))
        var += area_residual ** 2 * d_i / (n_i * (n_i - d_i))
    return area, float(var)


def comparar_rmst(df: pd.DataFrame, cfg: AnalysisConfig,
                  estrato: str | None = None,
                  tau: float | None = None,
                  referencia: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RMST por grupo e contrastes contra a categoria de referência.

    O contraste é a diferença de dias vividos (no hospital, até tau) e seu
    teste z usa a soma das variâncias, pois os grupos são independentes.
    """
    estrato = estrato or cfg.exposicao_principal
    tau = tau or cfg.tau_dias
    referencia = referencia or cfg.referencia_exposicao

    linhas = {}
    for nivel, g in df.groupby(estrato, observed=True):
        if len(g) < 20:
            continue
        m, v = rmst_km(g["tempo"].to_numpy(), (g["status"] == 1).astype(int).to_numpy(), tau)
        linhas[str(nivel)] = {"n": len(g), "RMST (dias)": m, "EP": np.sqrt(v),
                              "IC95% inf": m - 1.96 * np.sqrt(v),
                              "IC95% sup": m + 1.96 * np.sqrt(v)}
    tabela = pd.DataFrame(linhas).T

    contrastes = []
    if referencia in linhas:
        m0, v0 = linhas[referencia]["RMST (dias)"], linhas[referencia]["EP"] ** 2
        for nivel, val in linhas.items():
            if nivel == referencia:
                continue
            dif = val["RMST (dias)"] - m0
            ep = np.sqrt(val["EP"] ** 2 + v0)
            z = dif / ep if ep > 0 else np.nan
            from scipy import stats
            contrastes.append({"Comparação": f"{nivel} vs {referencia}",
                               "Δ RMST (dias)": dif, "EP": ep,
                               "IC95% inf": dif - 1.96 * ep,
                               "IC95% sup": dif + 1.96 * ep,
                               "p": 2 * (1 - stats.norm.cdf(abs(z)))})
    return tabela, pd.DataFrame(contrastes)


# =========================================================================== #
# 4. Fachada
# =========================================================================== #
def analise_km_completa(df: pd.DataFrame, cfg: AnalysisConfig,
                        estrato: str | None = None) -> ResultadoKM:
    """Executa o bloco não paramétrico completo e devolve tudo empacotado."""
    estrato = estrato or cfg.exposicao_principal
    curvas = ajustar_km(df, cfg, estrato)
    marcos = [0, 5, 10, 15, 20, 25, 30]
    risco = tabela_numeros_em_risco(df, estrato, [t for t in marcos if t <= cfg.tau_dias])
    glob, pw = testes_logrank(df, cfg, estrato)
    rmst_tab, rmst_ctr = comparar_rmst(df, cfg, estrato)

    resumo = []
    for nivel, g in df.groupby(estrato, observed=True):
        n = len(g)
        obitos = int((g["status"] == 1).sum())
        km = curvas.get(str(nivel))
        letalidade = np.nan
        if km is not None:
            letalidade = float(1 - km.predict(cfg.tau_dias))
        resumo.append({"Grupo": str(nivel), "n": n, "Óbitos": obitos,
                       "Letalidade bruta (%)": 100 * obitos / n,
                       f"Risco KM em {cfg.tau_dias:g}d (%)": 100 * letalidade,
                       "Permanência mediana (dias)": float(g["tempo"].median())})
    return ResultadoKM(curvas=curvas, tabela_risco=risco, logrank_global=glob,
                       logrank_pareado=pw, rmst=rmst_tab, rmst_contrastes=rmst_ctr,
                       resumo=pd.DataFrame(resumo))
