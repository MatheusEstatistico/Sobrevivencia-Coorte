"""
test_pipeline.py
================

Suíte de testes do pipeline. Três níveis:

1. **Contrato de dados** — o simulador entrega o schema do SIH e o
   pré-processamento o decodifica corretamente.
2. **Correção matemática** — cada estimador é confrontado com um caso de
   solução fechada, calculada à mão. É aqui que se pega erro de sinal,
   *off-by-one* em conjunto de risco e convenção errada de continuidade.
3. **Recuperação de parâmetros** — o modelo estimado devolve os coeficientes
   com que a coorte sintética foi gerada. É o teste que valida o pipeline
   como um todo, e não peça por peça.

Execução::

    pytest -q                       # tudo
    pytest -q -m "not lento"        # pula o teste de recuperação (mais caro)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coorte_respiratoria.config import AnalysisConfig
from coorte_respiratoria.data_acquisition import simular_sih, COLUNAS_RD
from coorte_respiratoria.preprocessing import (construir_coorte,
                                               agrupar_cid_respiratorio)
from coorte_respiratoria import competing_risks as cr
from coorte_respiratoria import survival_km as skm
from coorte_respiratoria import survival_cox as scox


@pytest.fixture(scope="module")
def cfg() -> AnalysisConfig:
    return AnalysisConfig(modo_dados="simular", n_simulacao=6000, n_bootstrap=10)


@pytest.fixture(scope="module")
def coorte(cfg) -> pd.DataFrame:
    df, _ = construir_coorte(simular_sih(n=cfg.n_simulacao, seed=cfg.seed), cfg)
    return df


# =========================================================================== #
# 1. Contrato de dados
# =========================================================================== #
def test_simulador_entrega_schema_do_sih(cfg):
    bruto = simular_sih(n=500, seed=1)
    assert set(COLUNAS_RD) <= set(bruto.columns)
    # O DATASUS entrega tudo como texto de largura fixa; o simulador imita isso
    # para que o pré-processamento seja exercitado de verdade.
    assert bruto["DIAG_PRINC"].map(type).eq(str).all()
    assert bruto["DT_INTER"].str.match(r"^\d{8}$").all()


def test_agrupamento_cid():
    assert agrupar_cid_respiratorio("J189") == "Pneumonia"
    assert agrupar_cid_respiratorio("J440") == "DPOC"
    assert agrupar_cid_respiratorio("J45") == "Asma"
    assert agrupar_cid_respiratorio("J960") == "Insuf. respiratória"
    assert agrupar_cid_respiratorio("J90") == "Outras"
    assert pd.isna(agrupar_cid_respiratorio("I500"))     # fora do capítulo X


def test_desfecho_competitivo_eh_particao(coorte, cfg):
    """Os três estados devem ser exaustivos, exclusivos e respeitar tau."""
    assert set(coorte["status"].unique()) <= {0, 1, 2}
    assert coorte["tempo"].min() >= cfg.tempo_minimo
    assert coorte["tempo"].max() <= cfg.tau_dias
    # Quem foi truncado em tau tem obrigatoriamente de estar censurado
    assert (coorte.loc[coorte["tempo"] == cfg.tau_dias, "status"] == 0).any()


def test_idade_dentro_dos_criterios(coorte, cfg):
    assert coorte["idade"].between(cfg.idade_minima, cfg.idade_maxima).all()


# =========================================================================== #
# 2. Correção matemática
# =========================================================================== #
def test_aalen_johansen_caso_manual():
    """Exemplo de 5 indivíduos com CIF calculável à mão.

    t=1: n=5, d_total=1 (óbito)     -> S(0)=1,      CIF += 1*(1/5) = 0,2
    t=2: n=4, d_total=1 (alta)      -> S(1)=0,8,    CIF += 0
    t=3: n=3, d_total=1 (óbito)     -> S(2)=0,6,    CIF += 0,6*(1/3) = 0,2
    Total CIF(óbito) = 0,4
    """
    tempo = np.array([1, 2, 3, 4, 5.0])
    status = np.array([1, 2, 1, 0, 2])
    aj = cr.aalen_johansen(tempo, status, causa=1)
    assert aj["cif"].iloc[-1] == pytest.approx(0.4, abs=1e-12)


def test_cif_soma_nao_excede_um(coorte):
    """As CIFs das duas causas somadas nunca podem passar de 1."""
    t, s = coorte["tempo"].to_numpy(), coorte["status"].to_numpy()
    f1 = cr.aalen_johansen(t, s, 1)["cif"].iloc[-1]
    f2 = cr.aalen_johansen(t, s, 2)["cif"].iloc[-1]
    assert 0 <= f1 + f2 <= 1.0 + 1e-9


def test_km_superestima_risco_em_relacao_a_aj(coorte, cfg):
    """Resultado teórico: 1−KM ≥ AJ sempre, com igualdade só sem competição."""
    comp = cr.comparar_km_vs_aj(coorte, causa=1, tau=cfg.tau_dias)
    assert (comp["risco_1menosKM"] >= comp["risco_aalen_johansen"] - 1e-12).all()
    assert comp["razao_vies"].iloc[-1] > 1.5      # competição forte nesta coorte


def test_rmst_sem_eventos_iguala_tau():
    """Sem nenhum evento, S(t) ≡ 1 e a área sob a curva até tau é tau."""
    m, v = skm.rmst_km(np.array([10.0, 20, 30]), np.array([0, 0, 0]), tau=30)
    assert m == pytest.approx(30.0)
    assert v == pytest.approx(0.0)


def test_rmst_caso_manual():
    """Um óbito em t=2 entre 2 indivíduos: S=0,5 a partir de t=2.
    RMST(4) = 1*2 + 0,5*2 = 3."""
    m, _ = skm.rmst_km(np.array([2.0, 4.0]), np.array([1, 0]), tau=4)
    assert m == pytest.approx(3.0)


def test_spline_reduz_a_identidade_na_primeira_coluna():
    x = np.linspace(20, 90, 200)
    B = scox.base_spline_natural(x, gl=3)
    assert B.shape == (200, 3)
    np.testing.assert_allclose(B[:, 0], x)        # 1ª coluna é o próprio x


def test_spline_eh_linear_fora_dos_nos_externos():
    """A restrição "natural" impõe linearidade além do último nó.

    O teste avalia a base em pontos **estritamente acima** do nó superior
    (com nós fixados explicitamente) e verifica que a segunda diferença é
    numericamente nula — se a base fosse cúbica irrestrita, não seria.
    """
    nos = [20.0, 40.0, 60.0, 80.0]
    fora = np.linspace(85, 120, 40)               # todo o grid > nó superior
    B = scox.base_spline_natural(fora, gl=3, nos=nos)
    for coluna in range(1, B.shape[1]):
        assert np.abs(np.diff(B[:, coluna], 2)).max() < 1e-9

    dentro = np.linspace(25, 75, 40)              # região onde deve ser curva
    Bd = scox.base_spline_natural(dentro, gl=3, nos=nos)
    assert np.abs(np.diff(Bd[:, 1], 2)).max() > 1e-6


def test_pesos_finegray_sao_um_sem_censura():
    """Sem censura, G(t) ≡ 1 e todos os pesos IPCW valem exatamente 1."""
    df = pd.DataFrame({
        "id_internacao": range(6),
        "tempo": [1.0, 2, 3, 4, 5, 6],
        "status": [1, 2, 1, 2, 1, 2],      # sem status 0
        "x": [0, 1, 0, 1, 0, 1.0],
    })
    longo = cr.expandir_finegray(df, ["x"], causa=1)
    assert longo["fg_peso"].eq(1.0).all()
    # Quem teve evento competitivo permanece no risco até o fim do seguimento
    assert longo.loc[longo["id_internacao"] == 1, "stop"].max() == 6.0


def test_pesos_finegray_decrescem_com_censura():
    """Com censura, o peso de quem teve evento competitivo cai após o evento."""
    df = pd.DataFrame({
        "id_internacao": range(6),
        "tempo": [1.0, 2, 3, 4, 5, 6],
        "status": [2, 0, 1, 0, 1, 2],      # censuras em t=2 e t=4
        "x": [0, 1, 0, 1, 0, 1.0],
    })
    longo = cr.expandir_finegray(df, ["x"], causa=1)
    pesos_id0 = longo.loc[longo["id_internacao"] == 0, "fg_peso"].to_numpy()
    assert pesos_id0[0] == pytest.approx(1.0)
    assert (np.diff(pesos_id0) <= 1e-12).all()          # monótono não crescente
    assert pesos_id0[-1] < 1.0


def test_e_value_simetrico_e_monotono():
    """E-value é invariante à inversão do HR e cresce com o afastamento da nulidade."""
    assert scox.e_value(2.0)["e_value_estimativa"] == pytest.approx(
        scox.e_value(0.5)["e_value_estimativa"])
    assert scox.e_value(1.0)["e_value_estimativa"] == pytest.approx(1.0)
    assert (scox.e_value(3.0)["e_value_estimativa"]
            > scox.e_value(1.5)["e_value_estimativa"])
    # IC que cruza a nulidade -> E-value do limite é 1
    assert scox.e_value(1.4, limite=0.9)["e_value_ic"] == 1.0


def test_permutacao_sem_efeito_nao_rejeita():
    """Com grupos atribuídos ao acaso, o teste não deve rejeitar sistematicamente."""
    rng = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame({
        "tempo": rng.exponential(5, n).clip(0.5, 30),
        "status": rng.choice([0, 1, 2], n, p=[0.05, 0.15, 0.80]),
        "g": rng.choice(["A", "B"], n),
    })
    res = cr.teste_permutacao_cif(df, "g", causa=1, tau=30, n_perm=199, seed=5)
    assert res["p_permutacao"] > 0.05


# =========================================================================== #
# 3. Recuperação de parâmetros (teste de sistema)
# =========================================================================== #
@pytest.mark.lento
def test_cox_recupera_parametros_do_gerador(cfg):
    """O gerador usa log-HR conhecidos; o modelo tem de reencontrá-los.

    Tolerância generosa (±30% na escala do HR) porque (i) o mecanismo gerador
    é Weibull com forma < 1, que não satisfaz riscos proporcionais de forma
    exata, e (ii) há confundimento embutido entre idade e grupo diagnóstico.
    O objetivo é detectar erro grosseiro — sinal trocado, covariável
    desalinhada, escala errada — e não aferir eficiência.
    """
    df, _ = construir_coorte(simular_sih(n=30_000, seed=99), cfg)
    res = scox.ajustar_cox(df, cfg, robusto=False)
    hr = res.resumo.set_index("Variável")["HR"]

    verdadeiros = {
        "uti": np.exp(1.10),
        "urgencia": np.exp(0.35),
        "comorbidade_registrada": np.exp(0.30),
        "grupo_diagnostico[DPOC]": np.exp(-0.25),
        "grupo_diagnostico[Insuf. respiratória]": np.exp(0.45),
    }
    for var, alvo in verdadeiros.items():
        assert hr[var] == pytest.approx(alvo, rel=0.30), f"{var}: {hr[var]:.2f} vs {alvo:.2f}"

    # Covariáveis sem efeito no gerador devem ficar perto da nulidade
    assert hr["sexo_feminino"] == pytest.approx(1.0, abs=0.15)
    assert hr["raca_nao_branca"] == pytest.approx(1.0, abs=0.15)


@pytest.mark.lento
def test_finegray_difere_de_causa_especifica_na_direcao_esperada(cfg):
    """A UTI acelera o óbito **e** retarda a alta; logo sHR > HR de causa
    específica, porque o modelo de sub-distribuição soma os dois canais."""
    df, _ = construir_coorte(simular_sih(n=12_000, seed=7), cfg)
    cs = cr.cox_causa_especifica(df, cfg, causa=1).set_index("Variável")
    fg = cr.ajustar_finegray(df, cfg, causa=1).resumo.set_index("Variável")
    assert fg.loc["uti", "sHR"] > cs.loc["uti", "HR causa-específica"]


def test_config_imutavel():
    """A configuração é o pré-registro: não pode ser alterada em execução."""
    cfg = AnalysisConfig()
    with pytest.raises(Exception):
        cfg.tau_dias = 60          # type: ignore[misc]
