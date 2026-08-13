"""
viz.py
======

Figuras em padrão de publicação (300 dpi, vetorial opcional, tipografia
consistente, paleta segura para daltonismo).

Princípios de design adotados
-----------------------------
1. **Tabela de números sob risco** sob toda curva de sobrevida. Sem ela, o
   leitor não distingue um platô real de um artefato de 4 pacientes.
2. **Bandas de confiança** desenhadas como *step* — a curva é uma função
   escada, e interpolar linearmente sugere uma suavidade que o estimador não
   possui.
3. **Paleta Okabe-Ito**, desenhada para ser distinguível nas formas mais comuns
   de daltonismo (~8% dos homens). Cor nunca é o único canal: usamos também
   estilo de linha.
4. **Sem eixo secundário, sem 3D, sem gradiente decorativo.** Razão de tinta
   sobre dados alta (Tufte).
5. Eixo y das curvas de mortalidade **não** é forçado a [0, 1]: com letalidade
   de ~8%, ancorar em 1 desperdiça 92% da área da figura. O corte é anotado
   explicitamente para não induzir exagero visual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Paleta Okabe-Ito (segura para deuteranopia/protanopia)
PALETA = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
ESTILOS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def aplicar_tema() -> None:
    """Configura o `rcParams` global. Chamar uma vez, no início do pipeline."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.9,
        "figure.constrained_layout.use": False,
    })


def _pct(x, _pos=None) -> str:
    return f"{100 * x:.0f}%"


def _salvar(fig: plt.Figure, caminho: Path, formatos: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Salva em PNG (rascunho/README) e PDF vetorial (submissão)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    saidas = []
    for f in formatos:
        p = caminho.with_suffix(f".{f}")
        fig.savefig(p, format=f)
        saidas.append(p)
    plt.close(fig)
    return saidas


# =========================================================================== #
# Figura 1 — Kaplan-Meier estratificado + tabela de risco
# =========================================================================== #
def figura_km(curvas: Mapping[str, object], tabela_risco: pd.DataFrame,
              caminho: Path, titulo: str,
              ylabel: str = "Probabilidade de permanecer internado e vivo",
              tau: float = 30.0, mostrar_ic: bool = True,
              p_logrank: float | None = None,
              anotacao: str | None = None) -> list[Path]:
    """Curvas de KM com bandas de confiança e tabela de números sob risco."""
    fig = plt.figure(figsize=(8.2, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.08)
    ax, ax_t = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for i, (nome, km) in enumerate(curvas.items()):
        cor, estilo = PALETA[i % len(PALETA)], ESTILOS[i % len(ESTILOS)]
        sf = km.survival_function_
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=cor, ls=estilo, label=nome)
        if mostrar_ic:
            ci = km.confidence_interval_
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                            step="post", color=cor, alpha=0.13, linewidth=0)

    ax.set_xlim(0, tau)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, loc="left")
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.legend(loc="lower left", ncols=2)
    ax.tick_params(labelbottom=False)

    texto = []
    if p_logrank is not None:
        texto.append("Log-rank: p < 0,001" if p_logrank < 1e-3 else f"Log-rank: p = {p_logrank:.3f}")
    if anotacao:
        texto.append(anotacao)
    if texto:
        ax.text(0.985, 0.06, "\n".join(texto), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", lw=0.7))

    # --------------------------- Tabela de risco -------------------------- #
    # A tabela vive em um eixo próprio que **compartilha a escala x** do painel
    # superior: assim cada número fica exatamente sob o ponto da curva a que se
    # refere, sem depender de coordenadas relativas (que desalinham quando a
    # figura muda de tamanho).
    tempos = [float(c) for c in tabela_risco.columns]
    n_lin = len(tabela_risco)

    ax_t.set_xlim(0, tau)
    ax_t.set_ylim(n_lin - 0.5, -0.9)                 # linha 0 no topo
    ax_t.set_yticks(range(n_lin))
    ax_t.set_yticklabels(tabela_risco.index, fontsize=8.8)
    for j, rot in enumerate(ax_t.get_yticklabels()):
        rot.set_color(PALETA[j % len(PALETA)])
    ax_t.set_xticks([t for t in tempos])
    ax_t.set_xlabel("Dias desde a internação")
    ax_t.tick_params(axis="y", length=0, pad=6)
    ax_t.grid(False)
    for lado in ("left", "top", "right"):
        ax_t.spines[lado].set_visible(False)
    ax_t.text(0, -0.8, "Nº sob risco", fontsize=9.3, fontweight="bold", va="center")

    for j, (_, linha) in enumerate(tabela_risco.iterrows()):
        for t, v in zip(tempos, linha.to_numpy()):
            # As colunas extremas são ancoradas pela borda, e não pelo centro:
            # centrado em t=0 metade do número cairia fora do eixo, sobrepondo
            # o rótulo do grupo.
            ha = "left" if t == tempos[0] else ("right" if t == tempos[-1] else "center")
            ax_t.text(t, j, f"{int(v):,}".replace(",", "."), ha=ha,
                      va="center", fontsize=8.2)
    return _salvar(fig, caminho)


# =========================================================================== #
# Figura 2 — Incidência acumulada (Aalen-Johansen)
# =========================================================================== #
def figura_cif(cifs: Mapping[str, pd.DataFrame], caminho: Path, titulo: str,
               ylabel: str = "Incidência acumulada de óbito intra-hospitalar",
               p_valor: float | None = None) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for i, (nome, d) in enumerate(cifs.items()):
        cor, estilo = PALETA[i % len(PALETA)], ESTILOS[i % len(ESTILOS)]
        ax.step(d["tempo"], d["cif"], where="post", color=cor, ls=estilo, label=nome)
        ax.fill_between(d["tempo"], d["ic_inf"], d["ic_sup"], step="post",
                        color=cor, alpha=0.13, linewidth=0)
    ax.set_xlabel("Dias desde a internação")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, loc="left")
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_xlim(0, max(d["tempo"].max() for d in cifs.values()))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    if p_valor is not None:
        # Um teste de permutação com B réplicas não distingue p abaixo de
        # 1/(B+1); reportar "p = 0,000" nesse caso seria falsa precisão.
        txt = (f"Teste de permutação: p = {p_valor:.3f}".replace(".", ",")
               if p_valor > 0.002 else "Teste de permutação: p < 0,002")
        ax.text(0.985, 0.05, txt,
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", lw=0.7))
    ax.text(0.985, 0.97, "Estimador de Aalen-Johansen\n(alta hospitalar como risco competitivo)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.4, color="0.35")
    return _salvar(fig, caminho)


# =========================================================================== #
# Figura 3 — Viés do Kaplan-Meier na presença de risco competitivo
# =========================================================================== #
def figura_km_vs_aj(comp: pd.DataFrame, caminho: Path) -> list[Path]:
    """Painel pedagógico: 1−KM vs Aalen-Johansen e a razão entre eles."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.step(comp["tempo"], comp["risco_1menosKM"], where="post", color=PALETA[1],
             ls="--", label="1 − Kaplan-Meier (alta tratada como censura)")
    ax1.step(comp["tempo"], comp["risco_aalen_johansen"], where="post", color=PALETA[0],
             label="Aalen-Johansen (estimador correto)")
    ax1.fill_between(comp["tempo"], comp["risco_aalen_johansen"], comp["risco_1menosKM"],
                     step="post", color=PALETA[1], alpha=0.14, linewidth=0,
                     label="Viés (superestimação)")
    ax1.set_ylabel("Risco acumulado de óbito")
    ax1.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax1.set_title("Tratar o evento competitivo como censura superestima o risco absoluto",
                  loc="left")
    ax1.legend(loc="upper left")

    ax2.plot(comp["tempo"], comp["razao_vies"], color="0.25")
    ax2.axhline(1.0, color=PALETA[2], lw=1.1, ls=":")
    ax2.set_ylabel("Razão\n(1−KM) / AJ")
    ax2.set_xlabel("Dias desde a internação")
    ax2.set_ylim(0.95, float(np.nanmax(comp["razao_vies"])) * 1.05 + 0.02)
    return _salvar(fig, caminho)


# =========================================================================== #
# Figura 4 — Forest plot
# =========================================================================== #
def figura_forest(tab: pd.DataFrame, caminho: Path, titulo: str,
                  col_estimativa: str = "HR", col_inf: str = "IC95% inf",
                  col_sup: str = "IC95% sup", col_rotulo: str = "Variável",
                  col_p: str = "p", rotulos_bonitos: Mapping[str, str] | None = None,
                  xlabel: str = "Razão de riscos (escala log)") -> list[Path]:
    """Forest plot em escala logarítmica com IC95% e coluna numérica à direita.

    Escala log é obrigatória: em escala linear, um HR de 0,5 e um de 2,0 —
    efeitos simétricos e de mesma magnitude — aparecem com distâncias
    visuais completamente diferentes da nulidade.
    """
    d = tab.copy().reset_index(drop=True)
    if rotulos_bonitos:
        d[col_rotulo] = d[col_rotulo].map(lambda x: rotulos_bonitos.get(x, x))
    d = d.iloc[::-1].reset_index(drop=True)      # topo = primeira linha da tabela
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(d) + 2.0))
    ax.axvline(1.0, color="0.45", lw=1.0, ls="--", zorder=1)
    for i, r in d.iterrows():
        signif = r[col_p] < 0.05 if col_p in d else True
        cor = PALETA[0] if signif else "0.55"
        ax.plot([r[col_inf], r[col_sup]], [y[i], y[i]], color=cor, lw=1.7, zorder=2)
        ax.plot([r[col_inf], r[col_inf]], [y[i] - .13, y[i] + .13], color=cor, lw=1.3)
        ax.plot([r[col_sup], r[col_sup]], [y[i] - .13, y[i] + .13], color=cor, lw=1.3)
        ax.scatter(r[col_estimativa], y[i], s=44, color=cor, zorder=3,
                   marker="s" if signif else "o")

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(d[col_rotulo])
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, loc="left")
    ax.grid(axis="y", visible=False)
    # Em escala logarítmica o matplotlib às vezes rotula apenas a década;
    # fixamos marcas interpretáveis dentro do intervalo observado.
    candidatos = np.array([0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4, 6, 8, 12])
    lo, hi = float(d[col_inf].min()), float(d[col_sup].max())
    marcas = candidatos[(candidatos >= lo * 0.85) & (candidatos <= hi * 1.15)]
    if len(marcas) >= 3:
        ax.set_xticks(marcas)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
    ax.tick_params(axis="x", which="minor", length=0)

    lim = ax.get_xlim()
    x_txt = lim[1] * 1.35
    for i, r in d.iterrows():
        txt = f"{r[col_estimativa]:.2f} ({r[col_inf]:.2f}–{r[col_sup]:.2f})"
        if col_p in d:
            txt += "   p<0,001" if r[col_p] < 1e-3 else f"   p={r[col_p]:.3f}"
        ax.text(x_txt, y[i], txt.replace(".", ","), va="center", fontsize=8.6)
    ax.set_xlim(lim[0], lim[1])
    ax.text(x_txt, len(d) - 0.3, f"{col_estimativa} (IC95%)", fontsize=9, fontweight="bold")
    return _salvar(fig, caminho)


# =========================================================================== #
# Figura 5 — Efeito não linear da idade
# =========================================================================== #
def figura_spline_idade(curva: pd.DataFrame, caminho: Path,
                        idade_ref: float = 65.0) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(curva["idade"], curva["hr"], color=PALETA[0])
    ax.fill_between(curva["idade"], curva["ic_inf"], curva["ic_sup"],
                    color=PALETA[0], alpha=0.16, linewidth=0)
    ax.axhline(1.0, color="0.45", ls="--", lw=1.0)
    ax.axvline(idade_ref, color="0.45", ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel(f"HR de óbito vs {idade_ref:.0f} anos")
    ax.set_title("Efeito da idade modelado por spline cúbica restrita (3 g.l.)", loc="left")
    # Sem marcas explícitas, a escala log costuma rotular apenas a década (só o
    # "1"), tornando a figura ilegível quando o efeito varia dentro de uma
    # ordem de grandeza.
    candidatos = np.array([0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
    lo, hi = float(curva["ic_inf"].min()), float(curva["ic_sup"].max())
    marcas = candidatos[(candidatos >= lo * 0.9) & (candidatos <= hi * 1.1)]
    if len(marcas) >= 3:
        ax.set_yticks(marcas)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.text(0.02, 0.95, "Faixa sombreada: IC95%", transform=ax.transAxes,
            va="top", fontsize=8.6, color="0.35")
    return _salvar(fig, caminho)


# =========================================================================== #
# Figura 6 — Diagnóstico de riscos proporcionais (Schoenfeld)
# =========================================================================== #
def figura_schoenfeld(modelo, dados: pd.DataFrame, caminho: Path,
                      variaveis: Sequence[str], max_paineis: int = 6,
                      coluna_tempo: str = "tempo") -> list[Path]:
    """Resíduos de Schoenfeld escalonados vs tempo, com tendência LOWESS.

    Sob riscos proporcionais, os resíduos devem flutuar em torno de zero sem
    tendência: qualquer inclinação sistemática indica :math:`\\beta(t)` variando
    no tempo.
    """
    from lifelines.statistics import proportional_hazard_test
    import statsmodels.api as sm

    res = modelo.compute_residuals(dados, kind="scaled_schoenfeld")
    teste = proportional_hazard_test(modelo, dados, time_transform="km").summary

    # `compute_residuals` devolve uma linha por **evento**, indexada pelo índice
    # do dataframe de treino — não pelo tempo. Sem este mapeamento, o eixo x
    # exibiria números de linha (0…n) rotulados como dias, o que produziria um
    # gráfico visualmente convincente e completamente errado.
    tempos = dados.loc[res.index, coluna_tempo].to_numpy(float)

    variaveis = [v for v in variaveis if v in res.columns][:max_paineis]
    n = len(variaveis)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.1 * nrows), squeeze=False)

    for k, v in enumerate(variaveis):
        ax = axes[k // ncols][k % ncols]
        yv = res[v].to_numpy(float)
        ax.scatter(tempos, yv, s=6, alpha=0.18, color="0.4", edgecolors="none")
        try:
            sm_low = sm.nonparametric.lowess(yv, tempos, frac=0.5)
            ax.plot(sm_low[:, 0], sm_low[:, 1], color=PALETA[1], lw=2)
        except Exception:                              # noqa: BLE001
            pass
        ax.axhline(0, color=PALETA[0], ls="--", lw=1.0)
        try:
            p = float(teste.loc[v, "p"]) if v in teste.index else np.nan
            rot = "p < 0,001" if p < 1e-3 else f"p = {p:.3f}".replace(".", ",")
        except Exception:                              # noqa: BLE001
            rot = ""
        ax.set_title(f"{v}   ({rot})", fontsize=9.5, loc="left", fontweight="normal")
        ax.set_xlabel("Tempo (dias)")
        ax.set_ylabel("Resíduo escalonado")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()
    fig.suptitle("Diagnóstico de riscos proporcionais — resíduos de Schoenfeld",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _salvar(fig, caminho)


# =========================================================================== #
# Figura 7 — CIF empilhada (decomposição dos destinos)
# =========================================================================== #
def figura_cif_empilhada(df: pd.DataFrame, caminho: Path, tau: float = 30.0,
                         titulo: str = "Destino da internação ao longo do tempo") -> list[Path]:
    """Área empilhada: proporção internada / alta / óbito em cada instante.

    As três frações somam 1 por construção (probabilidades de estado do
    processo multi-estado), o que torna a figura autoexplicativa e imune à
    crítica de "curvas que ultrapassam 100%".
    """
    from .competing_risks import aalen_johansen
    t, s = df["tempo"].to_numpy(float), df["status"].to_numpy(int)
    grade = np.linspace(0, tau, 250)

    aj_ob = aalen_johansen(t, s, 1)
    aj_al = aalen_johansen(t, s, 2)
    f_ob = np.interp(grade, aj_ob["tempo"], aj_ob["cif"], left=0.0)
    f_al = np.interp(grade, aj_al["tempo"], aj_al["cif"], left=0.0)
    internado = np.clip(1 - f_ob - f_al, 0, 1)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.stackplot(grade, internado, f_al, f_ob,
                 labels=["Internado", "Alta hospitalar", "Óbito"],
                 colors=["#DCE6EF", PALETA[2], PALETA[1]], alpha=0.95, edgecolor="white", lw=0.3)
    ax.set_xlim(0, tau)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dias desde a internação")
    ax.set_ylabel("Proporção da coorte")
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_title(titulo, loc="left")
    ax.legend(loc="center right")
    ax.grid(False)
    return _salvar(fig, caminho)
