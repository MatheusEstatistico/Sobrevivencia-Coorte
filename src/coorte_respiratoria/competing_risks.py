"""
competing_risks.py
==================

Análise de **riscos competitivos**: incidência acumulada de Aalen-Johansen,
modelos de risco de causa específica e o modelo de sub-distribuição de
Fine & Gray (1999), aqui implementado do zero por ponderação do conjunto de
risco.

O problema
----------
Um paciente internado por causa respiratória sai do hospital vivo (alta) ou
morre. A alta **impede** a observação do óbito intra-hospitalar: não é censura
independente, é um **evento competitivo**. Consequências:

* :math:`1 - \\mathrm{KM}(t)` **superestima** a probabilidade de óbito, e o viés
  cresce com a incidência do evento competitivo — que aqui é enorme (>90% das
  internações terminam em alta). Em coortes hospitalares esse viés pode ser de
  várias vezes a magnitude do risco real.
* O estimador correto do risco absoluto é a **incidência acumulada (CIF)** de
  Aalen-Johansen:

  .. math::
     \\hat F_k(t) = \\sum_{t_j \\le t} \\hat S(t_{j-1})\\,
                    \\frac{d_{kj}}{n_j}

  onde :math:`\\hat S` é o KM de "qualquer saída". O fator :math:`\\hat S(t_{j-1})`
  é justamente o que falta ao KM ingênuo: só pode morrer no dia *t* quem ainda
  está internado no dia *t*.

Dois modelos, duas perguntas
----------------------------
====================  ==========================================  =====================
Modelo                Estima                                       Serve para
====================  ==========================================  =====================
Causa específica      :math:`h_1(t)` entre os **ainda internados** Etiologia/mecanismo
Fine-Gray (sub-dist.) efeito sobre :math:`F_1(t)` diretamente      Prognóstico/risco
                                                                   absoluto, decisão
====================  ==========================================  =====================

Eles respondem a perguntas diferentes e **não** são intercambiáveis; reportar
ambos é a recomendação corrente (Austin, Lee & Fine, 2016; Latouche et al.,
2013). Um mesmo fator pode elevar o hazard de causa específica e não alterar a
CIF, se também acelerar o evento competitivo.

Implementação de Fine-Gray
--------------------------
O `lifelines` não traz Fine-Gray. Em vez de depender de R, implementamos a
equivalência de Geskus (2011): o modelo de sub-distribuição é um modelo de Cox
ponderado no qual os indivíduos que sofrem o **evento competitivo permanecem no
conjunto de risco** após o evento, com peso decrescente

.. math::
   w_i(t) = \\frac{\\hat G(t)}{\\hat G(T_i)},\\qquad t > T_i,

sendo :math:`\\hat G` o Kaplan-Meier da distribuição de censura (IPCW). Isso
replica exatamente o que `survival::finegray()` faz em R, e o ajuste passa a
ser um Cox com intervalos (start, stop) e pesos — que o `lifelines` sabe fazer
via `CoxTimeVaryingFitter`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter

from .config import AnalysisConfig
from .survival_cox import montar_matriz

log = logging.getLogger(__name__)


# =========================================================================== #
# 1. Incidência acumulada (Aalen-Johansen)
# =========================================================================== #
def aalen_johansen(tempo: np.ndarray, status: np.ndarray,
                   causa: int = 1) -> pd.DataFrame:
    """Estimador de Aalen-Johansen da CIF, com tratamento exato de empates.

    Implementado diretamente da definição para (i) evitar o *jitter* aleatório
    que algumas bibliotecas aplicam a tempos empatados — inaceitável em dados
    de permanência hospitalar, medidos em dias inteiros, onde os empates são a
    regra — e (ii) manter o código auditável linha a linha.

    Returns
    -------
    DataFrame com colunas: tempo, n_risco, d_causa, d_total, km_geral, cif.
    """
    tempo = np.asarray(tempo, float)
    status = np.asarray(status, int)

    t_evt = np.unique(tempo[status > 0])
    if t_evt.size == 0:
        return pd.DataFrame(columns=["tempo", "n_risco", "d_causa", "d_total", "km_geral", "cif"])

    n_risco = np.array([(tempo >= t).sum() for t in t_evt], dtype=float)
    d_total = np.array([((tempo == t) & (status > 0)).sum() for t in t_evt], dtype=float)
    d_causa = np.array([((tempo == t) & (status == causa)).sum() for t in t_evt], dtype=float)

    # KM de "qualquer saída" (sobrevida global no estado 'internado')
    km = np.cumprod(1.0 - d_total / n_risco)
    km_anterior = np.concatenate(([1.0], km[:-1]))       # S(t_{j-1})

    cif = np.cumsum(km_anterior * d_causa / n_risco)
    return pd.DataFrame({"tempo": t_evt, "n_risco": n_risco, "d_causa": d_causa,
                         "d_total": d_total, "km_geral": km, "cif": cif})


def cif_ic_bootstrap(tempo: np.ndarray, status: np.ndarray, causa: int,
                     grade: np.ndarray, n_boot: int = 300,
                     seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """IC95% percentílico da CIF por bootstrap não paramétrico.

    Optamos por bootstrap em vez da variância delta de Aalen porque (a) a
    fórmula analítica é sensível a estratos com poucos eventos na cauda e
    (b) o intervalo percentílico respeita os limites [0, 1] sem transformação
    ad hoc. Custo: O(B·n log n), aceitável nesta escala.
    """
    rng = np.random.default_rng(seed)
    n = len(tempo)
    curvas = np.empty((n_boot, len(grade)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        aj = aalen_johansen(tempo[idx], status[idx], causa)
        curvas[b] = (np.interp(grade, aj["tempo"], aj["cif"], left=0.0)
                     if len(aj) else np.zeros(len(grade)))
    return np.percentile(curvas, 2.5, axis=0), np.percentile(curvas, 97.5, axis=0)


def cif_por_grupo(df: pd.DataFrame, estrato: str, causa: int = 1,
                  tau: float = 30.0, n_boot: int = 200,
                  seed: int = 1) -> dict[str, pd.DataFrame]:
    """CIF (com IC bootstrap) para cada nível do estrato, em grade comum."""
    grade = np.linspace(0, tau, 200)
    saida: dict[str, pd.DataFrame] = {}
    for nivel, g in df.groupby(estrato, observed=True):
        if len(g) < 20:
            continue
        t = g["tempo"].to_numpy(float)
        s = g["status"].to_numpy(int)
        aj = aalen_johansen(t, s, causa)
        cif = np.interp(grade, aj["tempo"], aj["cif"], left=0.0) if len(aj) else np.zeros_like(grade)
        lo, hi = cif_ic_bootstrap(t, s, causa, grade, n_boot=n_boot, seed=seed)
        saida[str(nivel)] = pd.DataFrame({"tempo": grade, "cif": cif,
                                          "ic_inf": lo, "ic_sup": hi})
    return saida


def comparar_km_vs_aj(df: pd.DataFrame, causa: int = 1,
                      tau: float = 30.0) -> pd.DataFrame:
    """Quantifica o viés de tratar o evento competitivo como censura.

    Retorna, na grade de tempo, o risco de óbito segundo (a) 1−KM, que ignora
    a competição, e (b) Aalen-Johansen. A razão entre eles é o fator de
    superestimação — um número que costuma surpreender revisores.
    """
    t = df["tempo"].to_numpy(float)
    s = df["status"].to_numpy(int)
    grade = np.linspace(0, tau, 200)

    # 1 - KM tratando alta como censura
    evt = (s == causa).astype(int)
    t_e = np.unique(t[evt == 1])
    if t_e.size:
        n_r = np.array([(t >= x).sum() for x in t_e], float)
        d = np.array([((t == x) & (evt == 1)).sum() for x in t_e], float)
        km = np.cumprod(1 - d / n_r)
        risco_km = np.interp(grade, t_e, 1 - km, left=0.0)
    else:
        risco_km = np.zeros_like(grade)

    aj = aalen_johansen(t, s, causa)
    risco_aj = np.interp(grade, aj["tempo"], aj["cif"], left=0.0) if len(aj) else np.zeros_like(grade)

    with np.errstate(divide="ignore", invalid="ignore"):
        razao = np.where(risco_aj > 0, risco_km / risco_aj, np.nan)
    return pd.DataFrame({"tempo": grade, "risco_1menosKM": risco_km,
                         "risco_aalen_johansen": risco_aj, "razao_vies": razao})


# =========================================================================== #
# 2. Teste de igualdade de CIFs (alternativa a Gray)
# =========================================================================== #
def teste_permutacao_cif(df: pd.DataFrame, estrato: str, causa: int = 1,
                         tau: float = 30.0, n_perm: int = 999,
                         seed: int = 7) -> dict[str, float]:
    """Teste de permutação para :math:`H_0: F_1^{(1)} = \\dots = F_1^{(k)}`.

    O teste k-amostral de Gray (1988) não está implementado nas bibliotecas
    Python de uso corrente. Como substituto exato-condicional, permutamos os
    rótulos de grupo (mantendo fixos os pares tempo-status, o que preserva a
    estrutura de censura e de competição) e usamos como estatística a soma dos
    desvios quadráticos das CIFs de grupo em relação à CIF combinada,
    integrada no tempo:

    .. math::
        T = \\sum_k n_k \\int_0^\\tau \\left(\\hat F_k(u) - \\hat F(u)\\right)^2 du

    O p-valor é :math:`(1 + \\#\\{T^{perm} \\ge T^{obs}\\}) / (1 + B)`, que é
    válido para qualquer B e nunca retorna zero exato.
    """
    rng = np.random.default_rng(seed)
    grade = np.linspace(0, tau, 100)
    t = df["tempo"].to_numpy(float)
    s = df["status"].to_numpy(int)
    g = df[estrato].astype(str).to_numpy()

    def estatistica(rotulos: np.ndarray) -> float:
        aj_all = aalen_johansen(t, s, causa)
        f_geral = np.interp(grade, aj_all["tempo"], aj_all["cif"], left=0.0)
        total = 0.0
        for nivel in np.unique(rotulos):
            m = rotulos == nivel
            aj = aalen_johansen(t[m], s[m], causa)
            f_k = np.interp(grade, aj["tempo"], aj["cif"], left=0.0) if len(aj) else np.zeros_like(grade)
            total += m.sum() * np.trapezoid((f_k - f_geral) ** 2, grade)
        return float(total)

    t_obs = estatistica(g)
    nulos = np.array([estatistica(rng.permutation(g)) for _ in range(n_perm)])
    p = (1 + int((nulos >= t_obs).sum())) / (1 + n_perm)
    return {"estatistica": t_obs, "p_permutacao": p, "n_permutacoes": n_perm}


# =========================================================================== #
# 3. Modelos de risco de causa específica
# =========================================================================== #
def cox_causa_especifica(df: pd.DataFrame, cfg: AnalysisConfig,
                         causa: int) -> pd.DataFrame:
    """Cox para a causa `causa`, tratando as demais causas como censura.

    Essa censura é legítima **aqui**: o alvo da estimação é o hazard de causa
    específica, definido condicionalmente a ainda estar sob risco. O que não se
    pode fazer é transformar :math:`\\hat h_1` em risco absoluto ignorando as
    outras causas.
    """
    X, cols = montar_matriz(df, cfg)
    X["evento"] = (df.loc[X.index, "status"] == causa).astype(int)
    X = X.dropna(subset=cols + ["tempo", "evento"])

    cph = CoxPHFitter().fit(X[cols + ["tempo", "evento"]], "tempo", "evento")
    s = cph.summary
    return pd.DataFrame({"Variável": s.index, "HR causa-específica": np.exp(s["coef"]),
                        "IC95% inf": np.exp(s["coef lower 95%"]),
                        "IC95% sup": np.exp(s["coef upper 95%"]),
                        "p": s["p"]}).reset_index(drop=True)


# =========================================================================== #
# 4. Fine-Gray por ponderação do conjunto de risco
# =========================================================================== #
def km_censura(tempo: np.ndarray, status: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier da distribuição de **censura** (KM reverso).

    Trocam-se os papéis: o "evento" passa a ser a censura administrativa/por
    transferência. Retorna os tempos de salto e o valor de G logo após cada
    salto (versão contínua à direita).
    """
    tempo = np.asarray(tempo, float)
    cens = (np.asarray(status, int) == 0).astype(int)
    t_c = np.unique(tempo[cens == 1])
    if t_c.size == 0:
        return np.array([np.inf]), np.array([1.0])
    n_r = np.array([(tempo >= x).sum() for x in t_c], float)
    d_c = np.array([((tempo == x) & (cens == 1)).sum() for x in t_c], float)
    G = np.cumprod(1.0 - d_c / n_r)
    return t_c, G


def expandir_finegray(df: pd.DataFrame, cols_covar: list[str],
                      causa: int = 1, id_col: str = "id_internacao",
                      tempo_col: str = "tempo",
                      status_col: str = "status") -> pd.DataFrame:
    """Expande a coorte no formato (start, stop] com pesos IPCW de Fine-Gray.

    Regras de expansão:

    * **Evento de interesse** em T: uma linha ``(0, T]``, evento = 1, peso = 1.
    * **Censurado** em C: uma linha ``(0, C]``, evento = 0, peso = 1.
    * **Evento competitivo** em T: o indivíduo permanece no conjunto de risco
      *depois* de T, com peso :math:`\\hat G(t)/\\hat G(T)`. Como :math:`\\hat G`
      só decresce nos tempos de censura, basta uma linha por tempo de censura
      posterior a T — daí o custo ser O(n · #tempos_de_censura) e não O(n · n).

    Verificações internas:
      - se não houver censura, todos os pesos valem 1 e a expansão é exata;
      - linhas com peso ≈ 0 (após G se anular) são descartadas por serem
        numericamente irrelevantes e potencialmente instáveis.
    """
    t = df[tempo_col].to_numpy(float)
    s = df[status_col].to_numpy(int)
    tempo_max = float(t.max())

    t_c, G = km_censura(t, s)
    t_c_validos = t_c[np.isfinite(t_c)]

    def G_em(x: float) -> float:
        """G(x) contínua à direita: valor após o último salto ≤ x."""
        if t_c_validos.size == 0:
            return 1.0
        pos = np.searchsorted(t_c_validos, x, side="right") - 1
        return 1.0 if pos < 0 else float(G[pos])

    linhas = []
    base = df[[id_col] + cols_covar].copy()
    base_dict = base.set_index(id_col).to_dict("index")

    for i in range(len(df)):
        uid = df[id_col].iloc[i]
        Ti, si = t[i], s[i]
        cov = base_dict[uid]

        if si == causa:                                   # evento de interesse
            linhas.append({**cov, id_col: uid, "start": 0.0, "stop": Ti,
                           "fg_evento": 1, "fg_peso": 1.0})
        elif si == 0:                                     # censurado
            linhas.append({**cov, id_col: uid, "start": 0.0, "stop": Ti,
                           "fg_evento": 0, "fg_peso": 1.0})
        else:                                             # evento competitivo
            G_Ti = G_em(Ti)
            saltos = t_c_validos[t_c_validos > Ti]
            if G_Ti <= 0 or saltos.size == 0:
                linhas.append({**cov, id_col: uid, "start": 0.0,
                               "stop": max(Ti, tempo_max), "fg_evento": 0, "fg_peso": 1.0})
                continue
            # (0, c_1]: ainda peso 1, pois G não caiu entre Ti e c_1
            linhas.append({**cov, id_col: uid, "start": 0.0, "stop": float(saltos[0]),
                           "fg_evento": 0, "fg_peso": 1.0})
            bordas = np.append(saltos, tempo_max)
            for k in range(len(saltos)):
                a, b = float(saltos[k]), float(bordas[k + 1])
                if b <= a:
                    continue
                peso = G_em(a) / G_Ti
                if peso < 1e-8:
                    break
                linhas.append({**cov, id_col: uid, "start": a, "stop": b,
                               "fg_evento": 0, "fg_peso": float(peso)})

    longo = pd.DataFrame(linhas)
    longo = longo[longo["stop"] > longo["start"]].reset_index(drop=True)
    return longo


@dataclass
class ResultadoFineGray:
    resumo: pd.DataFrame
    modelo: CoxTimeVaryingFitter
    n_linhas_expandidas: int
    peso_medio: float
    diagnostico: dict = field(default_factory=dict)


def ajustar_finegray(df: pd.DataFrame, cfg: AnalysisConfig,
                     causa: int = 1,
                     amostra_max: int | None = 25_000,
                     n_boot_ic: int = 0) -> ResultadoFineGray:
    """Ajusta o modelo de sub-distribuição de Fine-Gray.

    Parameters
    ----------
    amostra_max : int | None
        Teto de indivíduos antes da expansão. A expansão multiplica as linhas
        pelo número de tempos de censura distintos; com permanência medida em
        dias e tau = 30, o fator é ≤ 31, mas em coortes muito grandes convém
        subamostrar (a subamostragem é aleatória simples e não enviesa as
        estimativas, apenas as torna menos precisas).

    Interpretação
    -------------
    ``exp(beta)`` é a **razão de hazards de sub-distribuição (sHR)**. Um sHR de
    1,5 significa que a covariável está associada a uma CIF de óbito 
    mais alta; ao contrário do HR de causa específica, ela já incorpora o efeito
    da covariável sobre o evento competitivo.
    """
    X, cols = montar_matriz(df, cfg)
    dados = df.loc[X.index].copy()
    for c in cols:
        dados[c] = X[c]
    dados = dados.dropna(subset=cols + ["tempo", "status"])

    if amostra_max and len(dados) > amostra_max:
        dados = dados.sample(amostra_max, random_state=cfg.seed)
        log.info("Fine-Gray: subamostra de %d internações para a expansão.", amostra_max)

    longo = expandir_finegray(dados, cols, causa=causa, id_col=cfg.id_col)
    ctv = _fit_ctv(longo, cols, cfg)

    s = ctv.summary
    resumo = pd.DataFrame({
        "Variável": s.index,
        "sHR": np.exp(s["coef"]),
        "IC95% inf": np.exp(s["coef lower 95%"]),
        "IC95% sup": np.exp(s["coef upper 95%"]),
        "p": s["p"],
    }).reset_index(drop=True)

    diag = {"n_individuos": int(dados.shape[0]),
            "fator_expansao": len(longo) / max(len(dados), 1),
            "variancia": "modelo (naive)"}

    # ------------------------------------------------------------------ #
    # IC por bootstrap (opcional). A variância "naive" do Cox ponderado
    # ignora que os pesos IPCW foram **estimados** a partir dos próprios
    # dados; Fine & Gray (1999) derivam um sanduíche que corrige isso, e o
    # `survival::coxph` do R o obtém com `robust=TRUE, cluster=id`. O
    # `lifelines` ainda não implementa variância robusta em modelos com
    # covariáveis dependentes do tempo (`NotImplementedError`), de modo que
    # o bootstrap por indivíduo é aqui a alternativa correta — apenas mais
    # cara. Na prática a diferença costuma ser pequena quando a censura é
    # leve, como nesta coorte (~5%).
    # ------------------------------------------------------------------ #
    if n_boot_ic and n_boot_ic > 0:
        rng = np.random.default_rng(cfg.seed)
        coefs = []
        ids = dados[cfg.id_col].to_numpy()
        for _ in range(n_boot_ic):
            escolha = rng.choice(ids, size=len(ids), replace=True)
            amostra = dados.set_index(cfg.id_col).loc[escolha].reset_index()
            amostra[cfg.id_col] = np.arange(len(amostra))   # ids únicos
            try:
                lb = expandir_finegray(amostra, cols, causa=causa, id_col=cfg.id_col)
                coefs.append(_fit_ctv(lb, cols, cfg).params_.reindex(s.index).to_numpy())
            except Exception:                               # noqa: BLE001
                continue
        if len(coefs) >= 20:
            M = np.vstack(coefs)
            resumo["IC95% inf"] = np.exp(np.percentile(M, 2.5, axis=0))
            resumo["IC95% sup"] = np.exp(np.percentile(M, 97.5, axis=0))
            diag["variancia"] = f"bootstrap percentílico (B={len(coefs)})"

    return ResultadoFineGray(resumo=resumo, modelo=ctv,
                             n_linhas_expandidas=len(longo),
                             peso_medio=float(longo["fg_peso"].mean()),
                             diagnostico=diag)


def _fit_ctv(longo: pd.DataFrame, cols: list[str], cfg: AnalysisConfig) -> CoxTimeVaryingFitter:
    """Ajusta o Cox com intervalos (start, stop] e pesos IPCW."""
    ctv = CoxTimeVaryingFitter(penalizer=1e-7)
    ctv.fit(longo[[cfg.id_col, "start", "stop", "fg_evento", "fg_peso"] + cols],
            id_col=cfg.id_col, event_col="fg_evento",
            start_col="start", stop_col="stop",
            weights_col="fg_peso", robust=False, show_progress=False)
    return ctv


def tabela_comparativa(cs: pd.DataFrame, fg: pd.DataFrame) -> pd.DataFrame:
    """Junta HR de causa específica e sHR lado a lado (Tabela 4 do artigo).

    Divergências entre as duas colunas não são erro: elas são a informação.
    Quando o sHR é menor que o HR de causa específica, a covariável também
    acelera a alta hospitalar, retirando o paciente do risco antes que o
    óbito se materialize.
    """
    t = cs.merge(fg, on="Variável", suffixes=(" (causa-específica)", " (Fine-Gray)"))
    l_cs, l_fg = np.log(t["HR causa-específica"]), np.log(t["sHR"])
    dif = np.abs(l_cs - l_fg)

    # A ordem dos testes importa: uma covariável com HR = 0,98 e sHR = 1,04
    # tem "sinais opostos" no papel, mas ambos os efeitos são nulos. Só faz
    # sentido falar em direção divergente quando os dois efeitos são não
    # triviais (|log HR| > 0,10, i.e. mais de 10% de variação no hazard).
    nao_trivial = (np.abs(l_cs) > 0.10) & (np.abs(l_fg) > 0.10)
    t["Divergência"] = np.select(
        [dif <= 0.20,
         nao_trivial & (np.sign(l_cs) != np.sign(l_fg)),
         dif > 0.20],
        ["Concordantes", "Direção oposta", "Magnitude divergente"],
        default="Concordantes")
    return t
