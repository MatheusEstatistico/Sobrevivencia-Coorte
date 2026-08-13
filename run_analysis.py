#!/usr/bin/env python
"""
run_analysis.py
===============

Orquestrador do estudo: executa, em ordem, todas as etapas do plano de análise
e materializa tabelas (CSV + Markdown) e figuras (PNG + PDF) em ``outputs/``.

Uso
---
::

    python run_analysis.py                       # padrões (simula se não houver rede)
    python run_analysis.py --modo pysus          # exige download real do DATASUS
    python run_analysis.py --anos 2022 2023 --uf MG --tau 30
    python run_analysis.py --config config/analysis_config.yaml
    python run_analysis.py --rapido              # menos bootstrap/permutações (CI)

Ordem das etapas (espelha a seção Métodos do artigo)
----------------------------------------------------
0. Configuração e semente
1. Aquisição (PySUS → cache → simulação)
2. Coorte analítica e fluxograma de elegibilidade
3. Descrição basal (Tabela 1, ausências, densidade de incidência)
4. Não paramétrica (Kaplan-Meier, log-rank, RMST)
5. Riscos competitivos (Aalen-Johansen, KM vs AJ, permutação)
6. Cox de causa específica (+ Schoenfeld, splines, bootstrap, E-value)
7. Fine-Gray (sub-distribuição) e confronto com causa específica
8. Relatório consolidado em Markdown
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coorte_respiratoria.config import AnalysisConfig                     # noqa: E402
from coorte_respiratoria.data_acquisition import obter_coorte              # noqa: E402
from coorte_respiratoria.preprocessing import (construir_coorte,           # noqa: E402
                                               formatar_fluxograma, GRUPOS_DIAG)
from coorte_respiratoria import descriptive as desc                        # noqa: E402
from coorte_respiratoria import survival_km as skm                         # noqa: E402
from coorte_respiratoria import survival_cox as scox                       # noqa: E402
from coorte_respiratoria import competing_risks as cr                      # noqa: E402
from coorte_respiratoria import viz                                        # noqa: E402

log = logging.getLogger("pipeline")

ROTULOS = {
    "idade": "Idade (anos)", "tempo": "Permanência até desfecho (dias)",
    "dias_uti": "Diárias de UTI", "sexo": "Sexo", "raca": "Raça/cor",
    "grupo_diagnostico": "Grupo diagnóstico", "carater": "Caráter da internação",
    "uti": "Uso de UTI", "urgencia": "Internação de urgência",
    "comorbidade_registrada": "Comorbidade registrada", "inverno": "Internação no inverno",
    "sexo_feminino": "Sexo feminino", "raca_nao_branca": "Raça/cor não branca",
    "alta_complexidade": "Alta complexidade",
    "idade_ns1": "Idade (spline, termo linear)",
    "idade_ns2": "Idade (spline, termo 2)", "idade_ns3": "Idade (spline, termo 3)",
    "grupo_diagnostico[DPOC]": "DPOC vs Pneumonia",
    "grupo_diagnostico[Asma]": "Asma vs Pneumonia",
    "grupo_diagnostico[Insuf. respiratória]": "Insuf. respiratória vs Pneumonia",
    "grupo_diagnostico[Outras]": "Outras respiratórias vs Pneumonia",
}


# --------------------------------------------------------------------------- #
def salvar_tabela(df: pd.DataFrame, cfg: AnalysisConfig, nome: str,
                  titulo: str, casas: int = 3) -> None:
    """Persiste em CSV (reanálise) e Markdown (leitura direta no GitHub)."""
    cfg.dir_tabelas.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.dir_tabelas / f"{nome}.csv", index=False, encoding="utf-8")
    md = df.copy()
    for c in md.select_dtypes("number").columns:
        md[c] = md[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{casas}f}")
    with open(cfg.dir_tabelas / f"{nome}.md", "w", encoding="utf-8") as fh:
        fh.write(f"### {titulo}\n\n{md.to_markdown(index=False)}\n")
    log.info("  → tabela salva: %s", nome)


def cabecalho(txt: str) -> None:
    log.info("\n%s\n%s", txt, "─" * len(txt))


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Coorte de internações respiratórias — SIH/SUS")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--uf", type=str, default=None)
    ap.add_argument("--anos", type=int, nargs="+", default=None)
    ap.add_argument("--tau", type=float, default=None, help="Horizonte de seguimento (dias)")
    ap.add_argument("--modo", choices=["auto", "pysus", "simular"], default=None)
    ap.add_argument("--n-sim", type=int, default=None, help="n da coorte simulada")
    ap.add_argument("--rapido", action="store_true", help="Menos reamostragens (CI/testes)")
    ap.add_argument("--verboso", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verboso else logging.INFO,
                        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                        datefmt="%H:%M:%S")

    # ---------------------------- 0. Configuração ------------------------- #
    cfg = AnalysisConfig.from_yaml(args.config) if args.config else AnalysisConfig()
    mudancas = {}
    if args.uf: mudancas["uf"] = args.uf
    if args.anos: mudancas["anos"] = tuple(args.anos)
    if args.tau: mudancas["tau_dias"] = args.tau
    if args.modo: mudancas["modo_dados"] = args.modo
    if args.n_sim: mudancas["n_simulacao"] = args.n_sim
    if args.rapido: mudancas["n_bootstrap"] = 25
    if mudancas:
        cfg = replace(cfg, **mudancas)
    cfg.preparar_diretorios()
    np.random.seed(cfg.seed)
    viz.aplicar_tema()

    n_perm = 99 if args.rapido else 499
    n_boot_cif = 60 if args.rapido else 200
    t0 = time.time()

    cabecalho("ETAPA 0 · Configuração do estudo")
    log.info("UF=%s | anos=%s | tau=%.0f d | modo=%s | semente=%d",
             cfg.uf, list(cfg.anos), cfg.tau_dias, cfg.modo_dados, cfg.seed)

    # ---------------------------- 1. Aquisição ---------------------------- #
    cabecalho("ETAPA 1 · Aquisição dos microdados do SIH/SUS")
    bruto = obter_coorte(cfg)
    origem = bruto.attrs.get("origem", "desconhecida")
    log.info("Origem: %s | %d registros brutos | %d colunas", origem, len(bruto), bruto.shape[1])
    if origem == "simulado":
        log.warning("⚠ Coorte SIMULADA: os números abaixo NÃO são estimativas "
                    "epidemiológicas reais. Use --modo pysus para dados do DATASUS.")

    # ------------------------ 2. Coorte analítica ------------------------- #
    cabecalho("ETAPA 2 · Construção da coorte e elegibilidade")
    coorte, fluxo = construir_coorte(bruto, cfg)
    print("\n" + formatar_fluxograma(fluxo) + "\n")
    (cfg.dir_tabelas / "fluxograma.txt").write_text(formatar_fluxograma(fluxo), encoding="utf-8")
    coorte.to_parquet(cfg.dir_saidas / "coorte_analitica.parquet", index=False)

    n_ob = int((coorte.status == 1).sum())
    log.info("n = %d | óbitos = %d (%.2f%%) | altas = %d | censuras = %d",
             len(coorte), n_ob, 100 * n_ob / len(coorte),
             int((coorte.status == 2).sum()), int((coorte.status == 0).sum()))

    # --------------------------- 3. Descritiva ---------------------------- #
    cabecalho("ETAPA 3 · Caracterização basal (Tabela 1)")
    t1 = desc.tabela_um(
        coorte, grupo="grupo_diagnostico",
        continuas=["idade", "tempo", "dias_uti"],
        categoricas=["sexo", "raca", "carater", "uti", "comorbidade_registrada", "inverno"],
        referencia=cfg.referencia_exposicao, rotulos=ROTULOS)
    salvar_tabela(t1, cfg, "tabela1_caracteristicas", "Tabela 1 — Características basais")

    aus = desc.relatorio_ausencia(coorte, ["raca_nao_branca", "sexo_feminino", "idade", "uti"])
    salvar_tabela(aus, cfg, "tabela_s1_dados_ausentes", "Tabela S1 — Dados ausentes", 2)

    inc = desc.tabela_incidencia(coorte, "grupo_diagnostico", cfg.tau_dias)
    salvar_tabela(inc, cfg, "tabela2_densidade_incidencia",
                  "Tabela 2 — Densidade de incidência de óbito", 2)
    print(inc.to_string(index=False))

    # ------------------- 4. Não paramétrica (Kaplan-Meier) ---------------- #
    cabecalho("ETAPA 4 · Kaplan-Meier estratificado, log-rank e RMST")
    res_km = skm.analise_km_completa(coorte, cfg)
    salvar_tabela(res_km.resumo, cfg, "tabela3_km_resumo", "Tabela 3 — Desfechos por grupo", 2)
    salvar_tabela(res_km.logrank_global, cfg, "tabela_s2_logrank_global",
                  "Tabela S2 — Log-rank global", 4)
    salvar_tabela(res_km.logrank_pareado.reset_index(drop=True), cfg,
                  "tabela_s3_logrank_pareado",
                  "Tabela S3 — Log-rank par a par (Holm)", 4)
    salvar_tabela(res_km.rmst.reset_index(names="Grupo"), cfg, "tabela4_rmst",
                  f"Tabela 4 — RMST em {cfg.tau_dias:.0f} dias", 3)
    salvar_tabela(res_km.rmst_contrastes, cfg, "tabela4b_rmst_contrastes",
                  "Tabela 4b — Contrastes de RMST", 4)

    p_lr = float(res_km.logrank_global["p"].iloc[0])
    log.info("Log-rank global: χ² = %.1f (gl = %d), p = %.3g",
             res_km.logrank_global["estatistica_qui2"].iloc[0],
             int(res_km.logrank_global["gl"].iloc[0]), p_lr)

    viz.figura_km(res_km.curvas, res_km.tabela_risco,
                  cfg.dir_figuras / "fig1_kaplan_meier_grupos",
                  titulo="Figura 1 · Sobrevida livre de óbito intra-hospitalar por grupo "
                         f"diagnóstico ({cfg.uf}, {min(cfg.anos)}–{max(cfg.anos)})",
                  ylabel="Sobrevida estimada (Kaplan-Meier)",
                  tau=cfg.tau_dias, p_logrank=p_lr,
                  anotacao="Alta hospitalar tratada como censura\n(ver Figura 3)")

    curvas_perm = skm.ajustar_km(coorte, cfg, evento_alvo=-1)
    viz.figura_km(curvas_perm, res_km.tabela_risco,
                  cfg.dir_figuras / "fig2_permanencia_hospitalar",
                  titulo="Figura 2 · Tempo até a saída do hospital (óbito ou alta)",
                  ylabel="Probabilidade de permanecer internado", tau=cfg.tau_dias)

    # ----------------------- 5. Riscos competitivos ----------------------- #
    cabecalho("ETAPA 5 · Riscos competitivos (Aalen-Johansen)")
    comp = cr.comparar_km_vs_aj(coorte, causa=1, tau=cfg.tau_dias)
    razao_final = float(comp["razao_vies"].iloc[-1])
    log.info("Em t = %.0f d: 1−KM = %.2f%% vs AJ = %.2f%% (superestimação de %.0f%%)",
             cfg.tau_dias, 100 * comp["risco_1menosKM"].iloc[-1],
             100 * comp["risco_aalen_johansen"].iloc[-1], 100 * (razao_final - 1))
    salvar_tabela(comp.iloc[::20], cfg, "tabela_s4_km_vs_aj",
                  "Tabela S4 — Comparação 1−KM vs Aalen-Johansen", 4)
    viz.figura_km_vs_aj(comp, cfg.dir_figuras / "fig3_vies_km_vs_aalen_johansen")

    cifs = cr.cif_por_grupo(coorte, "grupo_diagnostico", causa=1,
                            tau=cfg.tau_dias, n_boot=n_boot_cif, seed=cfg.seed)
    perm = cr.teste_permutacao_cif(coorte, "grupo_diagnostico", causa=1,
                                   tau=cfg.tau_dias, n_perm=n_perm, seed=cfg.seed)
    log.info("Teste de permutação das CIFs: p = %.4f (B = %d)",
             perm["p_permutacao"], perm["n_permutacoes"])
    viz.figura_cif(cifs, cfg.dir_figuras / "fig4_incidencia_acumulada",
                   titulo="Figura 4 · Incidência acumulada de óbito intra-hospitalar",
                   p_valor=perm["p_permutacao"])
    viz.figura_cif_empilhada(coorte, cfg.dir_figuras / "fig5_estados_empilhados",
                             tau=cfg.tau_dias)

    cif_30 = pd.DataFrame([{"Grupo": g, "CIF óbito em 30 d (%)": 100 * d["cif"].iloc[-1],
                            "IC95% inf": 100 * d["ic_inf"].iloc[-1],
                            "IC95% sup": 100 * d["ic_sup"].iloc[-1]}
                           for g, d in cifs.items()])
    salvar_tabela(cif_30, cfg, "tabela5_cif_30dias",
                  "Tabela 5 — Incidência acumulada em 30 dias", 2)
    print(cif_30.to_string(index=False))

    # ---------------------------- 6. Cox ---------------------------------- #
    cabecalho("ETAPA 6 · Regressão de Cox (risco de causa específica)")
    univ = scox.cox_univariavel(coorte, cfg,
                                ["idade", "sexo_feminino", "raca_nao_branca", "uti",
                                 "urgencia", "comorbidade_registrada", "inverno"])
    salvar_tabela(univ, cfg, "tabela6a_cox_univariavel",
                  "Tabela 6a — Modelos de Cox univariáveis", 3)

    res_cox = scox.ajustar_cox(coorte, cfg, robusto=True)
    log.info("Modelo ajustado: n = %d, eventos = %d, C-index aparente = %.3f",
             res_cox.n, res_cox.eventos, res_cox.c_index)
    salvar_tabela(res_cox.resumo, cfg, "tabela6b_cox_multivariavel",
                  "Tabela 6b — Cox multivariável (SE robusto por hospital)", 4)

    if len(res_cox.ph):
        ph = res_cox.ph.reset_index()
        salvar_tabela(ph, cfg, "tabela_s5_schoenfeld",
                      "Tabela S5 — Teste de riscos proporcionais (Schoenfeld)", 4)
        if res_cox.ph_viola:
            log.warning("Riscos proporcionais violados em: %s", ", ".join(map(str, res_cox.ph_viola)))
            log.info("→ Reajustando com estratificação da(s) covariável(is) violadora(s).")

    # Modelo de sensibilidade estratificado por UTI (variável clássica de
    # não proporcionalidade: o efeito da UTI concentra-se nos primeiros dias)
    try:
        res_estrat = scox.ajustar_cox(coorte, cfg, estratos=["uti"], robusto=True)
        salvar_tabela(res_estrat.resumo, cfg, "tabela_s6_cox_estratificado",
                      "Tabela S6 — Cox estratificado por uso de UTI (sensibilidade)", 4)
    except Exception as exc:                              # noqa: BLE001
        log.warning("Modelo estratificado falhou: %s", exc)

    val = scox.validar_bootstrap(coorte, cfg)
    log.info("C-index aparente = %.3f | otimismo = %.4f | corrigido = %.3f (B = %d)",
             val["c_aparente"], val["otimismo"], val["c_corrigido"], val["n_reps_validas"])
    salvar_tabela(pd.DataFrame([val]), cfg, "tabela_s7_validacao_bootstrap",
                  "Tabela S7 — Validação interna (otimismo de Harrell)", 4)

    # E-values das estimativas da exposição principal
    ev = []
    for _, r in res_cox.resumo.iterrows():
        if str(r["Variável"]).startswith(cfg.exposicao_principal):
            lim = r["IC95% inf"] if r["HR"] > 1 else r["IC95% sup"]
            e = scox.e_value(float(r["HR"]), float(lim))
            ev.append({"Variável": ROTULOS.get(r["Variável"], r["Variável"]),
                       "HR": r["HR"], **e})
    if ev:
        salvar_tabela(pd.DataFrame(ev), cfg, "tabela_s8_evalues",
                      "Tabela S8 — E-values (confundimento não medido)", 2)

    # Os três termos da spline de idade não têm interpretação isolada como HR
    # (são uma base de funções); o efeito da idade é reportado na Figura 7.
    forest = res_cox.resumo[~res_cox.resumo["Variável"].str.startswith("idade_ns")].copy()
    viz.figura_forest(forest, cfg.dir_figuras / "fig6_forest_cox",
                      titulo="Figura 6 · Razões de risco ajustadas para óbito intra-hospitalar\n"
                             "(ajustado por idade em spline — efeito na Figura 7)",
                      rotulos_bonitos=ROTULOS)

    curva_idade = scox.curva_hr_idade(res_cox, coorte, idade_ref=65)
    if len(curva_idade):
        viz.figura_spline_idade(curva_idade, cfg.dir_figuras / "fig7_hr_idade_spline")

    try:
        viz.figura_schoenfeld(
            res_cox.modelo, res_cox.dados, cfg.dir_figuras / "fig8_schoenfeld",
            variaveis=[c for c in res_cox.covariaveis if not c.startswith("idade_ns")][:6])
    except Exception as exc:                              # noqa: BLE001
        log.warning("Figura de Schoenfeld não gerada: %s", exc)

    # -------------------------- 7. Fine-Gray ------------------------------ #
    cabecalho("ETAPA 7 · Modelo de sub-distribuição (Fine-Gray)")
    cs = cr.cox_causa_especifica(coorte, cfg, causa=1)
    salvar_tabela(cs, cfg, "tabela7a_causa_especifica",
                  "Tabela 7a — Cox de causa específica (óbito)", 4)

    cs_alta = cr.cox_causa_especifica(coorte, cfg, causa=2)
    salvar_tabela(cs_alta, cfg, "tabela7b_causa_especifica_alta",
                  "Tabela 7b — Cox de causa específica (alta hospitalar)", 4)

    try:
        fg = cr.ajustar_finegray(coorte, cfg, causa=1)
        log.info("Fine-Gray: %d linhas expandidas (fator %.1f×), peso médio = %.3f",
                 fg.n_linhas_expandidas, fg.diagnostico["fator_expansao"], fg.peso_medio)
        salvar_tabela(fg.resumo, cfg, "tabela7c_finegray",
                      "Tabela 7c — Modelo de sub-distribuição de Fine-Gray", 4)

        comparativo = cr.tabela_comparativa(cs, fg.resumo)
        salvar_tabela(comparativo, cfg, "tabela8_cs_vs_finegray",
                      "Tabela 8 — Causa específica vs Fine-Gray", 3)
        print(comparativo[["Variável", "HR causa-específica", "sHR", "Divergência"]]
              .to_string(index=False))

        fg_plot = fg.resumo[~fg.resumo["Variável"].str.startswith("idade_ns")]
        viz.figura_forest(fg_plot, cfg.dir_figuras / "fig9_forest_finegray",
                          titulo="Figura 9 · Razões de hazard de sub-distribuição (Fine-Gray)",
                          col_estimativa="sHR", rotulos_bonitos=ROTULOS,
                          xlabel="Razão de hazard de sub-distribuição (escala log)")
    except Exception as exc:                              # noqa: BLE001
        log.error("Fine-Gray falhou: %s", exc, exc_info=args.verboso)

    # ------------------------- 8. Relatório final ------------------------- #
    cabecalho("ETAPA 8 · Relatório consolidado")
    duracao = time.time() - t0
    meta = {
        "executado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duracao_s": round(duracao, 1),
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "origem_dados": origem,
        "n_coorte": int(len(coorte)),
        "n_obitos": n_ob,
        "config": cfg.to_dict(),
        "logrank_p": p_lr,
        "permutacao_cif_p": perm["p_permutacao"],
        "c_index_corrigido": val["c_corrigido"],
    }
    (cfg.dir_saidas / "metadados_execucao.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    linhas_md = [
        "# Resultados — Coorte de internações respiratórias (SIH/SUS)", "",
        f"*Gerado em {meta['executado_em']} · origem dos dados: **{origem}** · "
        f"tempo de execução: {duracao:.0f} s*", "",
        "> ⚠️ Se a origem for `simulado`, os números são de uma coorte sintética "
        "com parâmetros conhecidos e **não** têm validade epidemiológica.", "",
        "## Fluxograma de elegibilidade", "", "```", formatar_fluxograma(fluxo), "```", "",
        "## Síntese", "",
        f"- Coorte: **{len(coorte):,}** internações; **{n_ob:,}** óbitos "
        f"({100*n_ob/len(coorte):.2f}%).".replace(",", "."),
        f"- Log-rank global entre grupos diagnósticos: p = {p_lr:.3g}.",
        f"- Superestimação do risco por 1−KM em {cfg.tau_dias:.0f} dias: "
        f"**{100*(razao_final-1):.0f}%** acima da incidência acumulada de Aalen-Johansen.",
        f"- C-index corrigido por otimismo: **{val['c_corrigido']:.3f}**.", "",
        "## Tabelas", "",
    ]
    for p in sorted(cfg.dir_tabelas.glob("*.md")):
        linhas_md.append(f"- [`{p.stem}`]({p.relative_to(cfg.dir_saidas).as_posix()})")
    linhas_md += ["", "## Figuras", ""]
    for p in sorted(cfg.dir_figuras.glob("*.png")):
        linhas_md.append(f"### {p.stem}\n\n![{p.stem}]({p.relative_to(cfg.dir_saidas).as_posix()})\n")
    (cfg.dir_saidas / "RESULTADOS.md").write_text("\n".join(linhas_md), encoding="utf-8")

    log.info("Concluído em %.1f s. Saídas em %s", duracao, cfg.dir_saidas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
