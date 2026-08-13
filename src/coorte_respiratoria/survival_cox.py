"""
survival_cox.py
===============

Modelagem semiparamétrica de riscos proporcionais (Cox, 1972), com o conjunto
de verificações que uma submissão a periódico exige.

Modelo
------
Para o indivíduo *i* com covariáveis :math:`x_i`, a taxa instantânea de óbito
intra-hospitalar (causa específica) é

.. math::
    h_1(t \\mid x_i) = h_{0,1}(t)\\,\\exp(\\beta^\\top x_i)

O risco basal :math:`h_{0,1}(t)` fica não especificado; :math:`\\beta` é estimado
maximizando a verossimilhança parcial. Em dados hospitalares há muitos empates
(permanência medida em dias inteiros), então usamos a aproximação de **Efron**,
que é sensivelmente menos enviesada que a de Breslow sob empates numerosos —
é o padrão do `lifelines`.

Decisões de modelagem implementadas aqui
----------------------------------------
1. **Idade em spline cúbica restrita** (Harrell), não linear e não categorizada.
   Dicotomizar covariáveis contínuas descarta informação e cria confundimento
   residual (Royston, Altman & Sauerbrei, 2006).
2. **Erros-padrão robustos agrupados por hospital (CNES)**: pacientes do mesmo
   estabelecimento compartilham protocolos e perfil de gravidade; ignorar essa
   correlação intraclasse subestima a variância (sanduíche de Lin-Wei).
3. **Verificação de riscos proporcionais** por resíduos de Schoenfeld
   escalonados (Grambsch & Therneau, 1994) com correção multiplicidade; quando
   violada, a covariável migra para o termo de estratificação.
4. **Validação interna por bootstrap** (otimismo de Harrell) do C-index — a
   discriminação aparente é sempre otimista em dados de desenvolvimento.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

from .config import AnalysisConfig

log = logging.getLogger(__name__)


# =========================================================================== #
# 1. Spline cúbica restrita (Harrell)
# =========================================================================== #
def nos_padrao(x: np.ndarray, k: int) -> np.ndarray:
    """Percentis recomendados por Harrell (2015) para k nós."""
    tabela = {3: [.10, .50, .90], 4: [.05, .35, .65, .95],
              5: [.05, .275, .50, .725, .95], 6: [.05, .23, .41, .59, .77, .95]}
    q = tabela.get(k, np.linspace(.05, .95, k))
    return np.quantile(x[~np.isnan(x)], q)


def base_spline_natural(x: np.ndarray, gl: int = 3,
                        nos: Sequence[float] | None = None) -> np.ndarray:
    """Base de spline cúbica restrita (*restricted cubic spline*) com `gl` graus
    de liberdade, i.e. `gl + 1` nós.

    A restrição de linearidade além dos nós externos — que é o que diferencia a
    spline "natural"/restrita da cúbica irrestrita — evita o comportamento
    errático nas caudas, onde há poucos pacientes (aqui: > 95 anos).

    A primeira coluna é o próprio *x*; assim, testar :math:`H_0` de que as
    colunas 2..gl são nulas é um teste formal de **não linearidade** (Wald ou
    razão de verossimilhanças com gl−1 graus de liberdade).

    Returns
    -------
    ndarray (n, gl)
    """
    x = np.asarray(x, dtype=float)
    k = gl + 1
    t = np.asarray(nos if nos is not None else nos_padrao(x, k), dtype=float)
    if len(np.unique(t)) < k:                    # dados muito concentrados
        t = np.linspace(np.nanmin(x), np.nanmax(x), k)

    def cubo_pos(u: np.ndarray) -> np.ndarray:
        return np.where(u > 0, u ** 3, 0.0)

    denom = (t[-1] - t[0]) ** 2
    colunas = [x]
    for j in range(k - 2):
        termo = (cubo_pos(x - t[j])
                 - cubo_pos(x - t[-2]) * (t[-1] - t[j]) / (t[-1] - t[-2])
                 + cubo_pos(x - t[-1]) * (t[-2] - t[j]) / (t[-1] - t[-2]))
        colunas.append(termo / denom)
    base = np.column_stack(colunas)
    base.setflags(write=False)
    return np.array(base)


# =========================================================================== #
# 2. Matriz de delineamento
# =========================================================================== #
def montar_matriz(df: pd.DataFrame, cfg: AnalysisConfig,
                  incluir_exposicao: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Constrói a matriz de delineamento com *dummies* de tratamento.

    A categoria de referência da exposição é fixada em `cfg.referencia_exposicao`
    (Pneumonia), escolhida por ser a mais frequente e clinicamente mais
    interpretável como comparador.
    """
    cols: list[str] = []
    X = pd.DataFrame(index=df.index)

    if incluir_exposicao:
        cat = df[cfg.exposicao_principal].astype("category")
        for nivel in cat.cat.categories:
            if nivel == cfg.referencia_exposicao:
                continue
            nome = f"{cfg.exposicao_principal}[{nivel}]"
            X[nome] = (cat == nivel).astype(int)
            cols.append(nome)

    for c in cfg.covariaveis_ajuste:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
            cols.append(c)
        else:
            log.warning("Covariável ausente e ignorada: %s", c)

    X["tempo"] = df["tempo"].to_numpy()
    X["evento"] = (df["status"] == 1).astype(int).to_numpy()   # causa específica: óbito
    if cfg.cluster_col in df.columns:
        X[cfg.cluster_col] = df[cfg.cluster_col].to_numpy()
    return X, cols


# =========================================================================== #
# 3. Ajuste
# =========================================================================== #
@dataclass
class ResultadoCox:
    modelo: CoxPHFitter
    resumo: pd.DataFrame          # HR, IC95%, p — pronto para tabela do artigo
    ph: pd.DataFrame              # teste de Schoenfeld por covariável
    ph_viola: list[str]
    c_index: float
    c_index_corrigido: float | None
    n: int
    eventos: int
    dados: pd.DataFrame          # frame exatamente como passado ao .fit()
    covariaveis: list[str]


def ajustar_cox(df: pd.DataFrame, cfg: AnalysisConfig,
                estratos: Sequence[str] | None = None,
                robusto: bool = True,
                penalizador: float = 0.0) -> ResultadoCox:
    """Ajusta o modelo de Cox de causa específica para óbito intra-hospitalar.

    Parameters
    ----------
    estratos : lista de colunas de estratificação
        Covariáveis que violam riscos proporcionais entram aqui: cada estrato
        ganha seu próprio risco basal e **não** recebe um HR estimado — é o
        preço de admitir que seu efeito varia no tempo.
    penalizador : float
        Penalização L2 (ridge) na verossimilhança parcial. Útil quando há
        separação quase completa em subgrupos pequenos.
    """
    X, cols = montar_matriz(df, cfg)
    X = X.dropna(subset=cols + ["tempo", "evento"])

    dados = X[cols + ["tempo", "evento"]].copy()
    if estratos:
        for e in estratos:
            dados[e] = df.loc[X.index, e].astype(str).to_numpy()

    cluster = None
    if robusto and cfg.cluster_col in X.columns:
        dados[cfg.cluster_col] = X[cfg.cluster_col].to_numpy()
        cluster = cfg.cluster_col

    cph = CoxPHFitter(penalizer=penalizador)
    cph.fit(dados, duration_col="tempo", event_col="evento",
            strata=list(estratos) if estratos else None,
            cluster_col=cluster, robust=robusto)

    # --------------------------- Tabela do artigo ------------------------- #
    s = cph.summary
    resumo = pd.DataFrame({
        "Variável": s.index,
        "HR": np.exp(s["coef"]),
        "IC95% inf": np.exp(s["coef lower 95%"]),
        "IC95% sup": np.exp(s["coef upper 95%"]),
        "EP (robusto)": s["se(coef)"],
        "z": s["z"],
        "p": s["p"],
    }).reset_index(drop=True)

    # ---------------- Riscos proporcionais (Schoenfeld) ------------------- #
    try:
        teste = proportional_hazard_test(cph, dados, time_transform="km")
        ph = teste.summary.copy()
        ph["viola"] = ph["p"] < cfg.alpha
        viola = list(ph.index[ph["viola"]].get_level_values(0)
                     if isinstance(ph.index, pd.MultiIndex) else ph.index[ph["viola"]])
    except Exception as exc:                       # noqa: BLE001
        log.warning("Teste de Schoenfeld indisponível: %s", exc)
        ph, viola = pd.DataFrame(), []

    c_apar = cph.concordance_index_
    return ResultadoCox(modelo=cph, resumo=resumo, ph=ph, ph_viola=viola,
                        c_index=c_apar, c_index_corrigido=None,
                        n=len(dados), eventos=int(dados["evento"].sum()),
                        dados=dados, covariaveis=cols)


# =========================================================================== #
# 4. Análise univariável (Tabela 2, coluna "bruto")
# =========================================================================== #
def cox_univariavel(df: pd.DataFrame, cfg: AnalysisConfig,
                    variaveis: Sequence[str]) -> pd.DataFrame:
    """Ajusta um Cox por covariável.

    Nota metodológica: os HR brutos são apresentados por transparência, **não**
    para selecionar variáveis. Seleção por p-valor univariável (p < 0,20) é uma
    prática desaconselhada — enviesa coeficientes e ICs. O conjunto de ajuste
    aqui é definido a priori por conhecimento causal (DAG em `docs/ARTIGO.md`).
    """
    linhas = []
    for v in variaveis:
        d = pd.DataFrame({
            "x": pd.to_numeric(df[v], errors="coerce"),
            "tempo": df["tempo"],
            "evento": (df["status"] == 1).astype(int),
        }).dropna()
        if d["x"].nunique() < 2 or d["evento"].sum() < 5:
            continue
        m = CoxPHFitter().fit(d, "tempo", "evento")
        s = m.summary.loc["x"]
        linhas.append({"Variável": v, "HR bruto": np.exp(s["coef"]),
                       "IC95% inf": np.exp(s["coef lower 95%"]),
                       "IC95% sup": np.exp(s["coef upper 95%"]), "p": s["p"]})
    return pd.DataFrame(linhas)


# =========================================================================== #
# 5. Validação interna: otimismo de Harrell por bootstrap
# =========================================================================== #
def validar_bootstrap(df: pd.DataFrame, cfg: AnalysisConfig,
                      n_reps: int | None = None) -> dict[str, float]:
    """Corrige o C-index pelo **otimismo**.

    Procedimento (Harrell, Lee & Mark, 1996):

    1. C_aparente: ajusta e avalia o modelo na amostra original.
    2. Para b = 1..B: reamostra com reposição; ajusta o modelo na reamostra;
       calcula C_boot (na reamostra) e C_orig (na amostra original).
    3. Otimismo = média(C_boot − C_orig).
    4. C_corrigido = C_aparente − Otimismo.

    Diferentemente de uma divisão treino/teste única, o bootstrap usa 100% dos
    dados para desenvolvimento e tem menor variância — razão pela qual o TRIPOD
    o recomenda como validação interna preferencial.
    """
    n_reps = n_reps or cfg.n_bootstrap
    rng = np.random.default_rng(cfg.seed)

    X, cols = montar_matriz(df, cfg)
    X = X.dropna(subset=cols + ["tempo", "evento"])[cols + ["tempo", "evento"]]

    base = CoxPHFitter().fit(X, "tempo", "evento")
    c_aparente = base.concordance_index_

    otimismos = []
    for b in range(n_reps):
        idx = rng.integers(0, len(X), len(X))
        # `reset_index` é indispensável: a reamostragem com reposição cria
        # rótulos de índice duplicados e as rotinas internas do lifelines
        # alinham predições pelo índice, o que corromperia o C-index da
        # reamostra (bug silencioso que produz otimismo negativo).
        amostra = X.iloc[idx].reset_index(drop=True)
        if amostra["evento"].sum() < 10:
            continue
        try:
            m = CoxPHFitter(penalizer=1e-6).fit(amostra, "tempo", "evento")
        except Exception:                          # noqa: BLE001
            continue
        # Ambos os C-index são calculados pela mesma rotina, com o sinal do
        # preditor invertido (maior hazard = menor sobrevida esperada).
        c_boot = concordance_index(amostra["tempo"],
                                   -m.predict_partial_hazard(amostra).to_numpy(),
                                   amostra["evento"])
        c_orig = concordance_index(X["tempo"],
                                   -m.predict_partial_hazard(X).to_numpy(),
                                   X["evento"])
        otimismos.append(c_boot - c_orig)

    otimismo = float(np.mean(otimismos)) if otimismos else 0.0
    return {"c_aparente": float(c_aparente),
            "otimismo": otimismo,
            "c_corrigido": float(c_aparente - otimismo),
            "n_reps_validas": len(otimismos)}


# =========================================================================== #
# 6. E-value (sensibilidade a confundimento não medido)
# =========================================================================== #
def e_value(hr: float, limite: float | None = None) -> dict[str, float]:
    """E-value de VanderWeele & Ding (2017).

    Responde: *qual seria a força mínima de associação (em RR) que um
    confundidor não medido precisaria ter — tanto com a exposição quanto com o
    desfecho — para explicar integralmente o HR observado?*

    Para desfechos relativamente raros, o HR aproxima o RR e a fórmula é
    :math:`E = RR + \\sqrt{RR(RR-1)}` (com RR → 1/RR se RR < 1).
    Um E-value alto indica que o achado é robusto; um E-value próximo de 1
    indica que um confundidor fraco bastaria para anulá-lo.
    """
    def _e(rr: float) -> float:
        rr = 1 / rr if rr < 1 else rr
        return rr + np.sqrt(rr * (rr - 1))

    saida = {"e_value_estimativa": _e(hr)}
    if limite is not None:
        # E-value do limite do IC mais próximo da nulidade
        saida["e_value_ic"] = 1.0 if (limite - 1) * (hr - 1) <= 0 else _e(limite)
    return saida


# =========================================================================== #
# 7. Efeito não linear da idade — para a figura de HR suavizado
# =========================================================================== #
def curva_hr_idade(res: ResultadoCox, df: pd.DataFrame,
                   idade_ref: float = 65.0) -> pd.DataFrame:
    """HR do óbito em função da idade, relativo a `idade_ref`, com IC95%.

    Usa a matriz de covariância dos coeficientes da spline para propagar a
    incerteza: :math:`\\mathrm{Var}(c^\\top\\hat\\beta) = c^\\top \\hat\\Sigma c`,
    onde *c* é o contraste entre a base spline na idade *a* e em `idade_ref`.
    """
    termos = [c for c in res.resumo["Variável"] if c.startswith("idade_ns")]
    if not termos:
        return pd.DataFrame()

    beta = res.modelo.params_[termos].to_numpy()
    Sigma = res.modelo.variance_matrix_.loc[termos, termos].to_numpy()

    grade = np.linspace(np.nanpercentile(df["idade"], 1),
                        np.nanpercentile(df["idade"], 99), 120)
    nos = nos_padrao(df["idade"].to_numpy(float), len(termos) + 1)
    B = base_spline_natural(grade, gl=len(termos), nos=nos)
    b_ref = base_spline_natural(np.array([idade_ref]), gl=len(termos), nos=nos)[0]

    contraste = B - b_ref
    log_hr = contraste @ beta
    ep = np.sqrt(np.einsum("ij,jk,ik->i", contraste, Sigma, contraste))
    return pd.DataFrame({"idade": grade, "hr": np.exp(log_hr),
                         "ic_inf": np.exp(log_hr - 1.96 * ep),
                         "ic_sup": np.exp(log_hr + 1.96 * ep)})
