"""
data_acquisition.py
===================

Camada de aquisição: obtenção das AIH (Autorizações de Internação Hospitalar)
do SIH/SUS via **PySUS**, com cache local em Parquet e um gerador sintético de
contingência.

Por que uma camada de aquisição isolada?
----------------------------------------
1. **Reprodutibilidade.** O FTP do DATASUS é instável e os arquivos são
   *retificados* retroativamente (uma AIH de janeiro pode mudar em uma
   competência posterior). Sem cache datado, a mesma análise roda hoje e não
   reproduz amanhã. Aqui todo download é materializado em Parquet com carimbo
   de tempo.
2. **Testabilidade.** O restante do pipeline nunca chama a rede. Ele consome um
   DataFrame com o *schema* da AIH-RD, venha ele do DATASUS ou do simulador.
   Isso permite CI (GitHub Actions) sem acesso ao FTP.
3. **Estabilidade de API.** O PySUS mudou de interface entre as versões maiores
   (0.x → 1.x → 2.x). O adaptador abaixo tenta as três assinaturas conhecidas,
   em ordem, e degrada com elegância.

Nota sobre o SIH/SUS
--------------------
O arquivo **RD** ("AIH Reduzida", `RDMGAAMM.dbc`) é o arquivo analítico do SIH:
uma linha por AIH encerrada, com diagnóstico principal (CID-10), datas de
internação e saída, dias de permanência, uso de UTI, caráter da internação e
o indicador de óbito. É um registro **administrativo de faturamento**, não um
prontuário — implicação metodológica discutida em `docs/ARTIGO.md`.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import AnalysisConfig

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Colunas da AIH-RD efetivamente usadas. Selecionar colunas na leitura reduz
# a memória em ~90% (o RD tem ~110 colunas e MG gera ~1,3 milhão de AIH/ano).
# --------------------------------------------------------------------------- #
COLUNAS_RD = [
    "N_AIH",       # identificador da AIH
    "IDENT",       # 1 = AIH normal, 5 = AIH de longa permanência (continuação)
    "CNES",        # estabelecimento (unidade de agrupamento para SE robusto)
    "MUNIC_RES",   # município de residência (IBGE, 6 dígitos)
    "MUNIC_MOV",   # município do estabelecimento
    "NASC",        # data de nascimento (AAAAMMDD)
    "SEXO",        # 1/M = masculino; 3/F = feminino
    "IDADE",       # idade na unidade indicada por COD_IDADE
    "COD_IDADE",   # 2 = dias, 3 = meses, 4 = anos, 5 = anos (>=100)
    "RACA_COR",    # 01 branca ... 05 indígena, 99 sem informação
    "DT_INTER",    # data de internação (AAAAMMDD)
    "DT_SAIDA",    # data de saída (AAAAMMDD)
    "DIAS_PERM",   # dias de permanência
    "MORTE",       # 1 = óbito na internação
    "COBRANCA",    # motivo de saída/permanência -> define o tipo de desfecho
    "DIAG_PRINC",  # CID-10 principal
    "DIAG_SECUN",  # CID-10 secundário
    "CAR_INT",     # caráter da internação (01 eletivo, 02 urgência, ...)
    "MARCA_UTI",   # tipo de leito de UTI utilizado (00 = nenhum)
    "UTI_MES_TO",  # total de diárias de UTI
    "ESPEC",       # especialidade do leito
    "COMPLEX",     # complexidade (02 média, 03 alta)
    "VAL_TOT",     # valor total da AIH (proxy de intensidade assistencial)
]


# =========================================================================== #
# 1. ADAPTADOR PySUS
# =========================================================================== #
def _baixar_pysus_v2(uf: str, ano: int, meses: Iterable[int], grupo: str) -> pd.DataFrame:
    """PySUS >= 2.0 — API funcional de alto nível.

    A partir da 2.x o PySUS expõe funções por sistema de informação e usa
    DuckDB por baixo, o que permite *predicate pushdown* via SQL: filtramos
    J00-J99 **no arquivo Parquet**, antes de materializar em memória.

    Equivalente em linha de comando::

        python -c "import pysus; pysus.sih(state='MG', year=2023, month=1, group='RD')"
    """
    import pysus  # import tardio: dependência opcional

    log.info("PySUS 2.x | sih(state=%s, year=%s, month=%s, group=%s)", uf, ano, list(meses), grupo)
    df = pysus.sih(
        state=uf,
        year=ano,
        month=list(meses),
        group=grupo,
        as_dataframe=True,
        show_progress=True,
        # Filtro empurrado para o motor de leitura (economiza RAM):
        sql="SELECT * FROM read_parquet WHERE DIAG_PRINC LIKE 'J%'",
    )
    return df


def _baixar_pysus_v1(uf: str, ano: int, meses: Iterable[int], grupo: str) -> pd.DataFrame:
    """PySUS 1.x — API orientada a objetos (`pysus.ftp.databases`)."""
    from pysus.ftp.databases.sih import SIH  # type: ignore

    sih = SIH().load()
    partes = []
    for mes in meses:
        arquivos = sih.get_files(grupo, uf=uf, year=ano, month=mes)
        if not arquivos:
            continue
        parquets = sih.download(arquivos)
        parquets = parquets if isinstance(parquets, list) else [parquets]
        for p in parquets:
            partes.append(p.to_dataframe())
    if not partes:
        raise RuntimeError("PySUS 1.x não retornou arquivos.")
    return pd.concat(partes, ignore_index=True)


def _baixar_pysus_v0(uf: str, ano: int, meses: Iterable[int], grupo: str) -> pd.DataFrame:
    """PySUS 0.x — API legada `online_data` (mantida para compatibilidade)."""
    from pysus.online_data.SIH import download  # type: ignore

    partes = [download(uf, ano, mes) for mes in meses]
    partes = [p for p in partes if p is not None and len(p)]
    if not partes:
        raise RuntimeError("PySUS 0.x não retornou dados.")
    return pd.concat(partes, ignore_index=True)


def baixar_sih(cfg: AnalysisConfig) -> pd.DataFrame:
    """Tenta baixar a AIH-RD pelas três gerações de API do PySUS, em ordem.

    Retorna o DataFrame **bruto** (colunas e códigos como o DATASUS entrega).
    Levanta `RuntimeError` se nenhuma estratégia funcionar — quem decide se
    isso é fatal é `obter_coorte`, conforme `cfg.modo_dados`.
    """
    erros: list[str] = []
    for nome, fn in (("2.x", _baixar_pysus_v2),
                     ("1.x", _baixar_pysus_v1),
                     ("0.x", _baixar_pysus_v0)):
        partes = []
        try:
            for ano in cfg.anos:
                partes.append(fn(cfg.uf, ano, cfg.meses, cfg.grupo_sih))
            df = pd.concat(partes, ignore_index=True)
            if len(df):
                log.info("Download concluído via PySUS %s: %d AIH.", nome, len(df))
                return df
            erros.append(f"{nome}: retornou 0 linhas")
        except Exception as exc:                      # noqa: BLE001
            erros.append(f"{nome}: {type(exc).__name__}: {exc}")
            log.warning("API PySUS %s indisponível (%s).", nome, exc)
    raise RuntimeError("Falha em todas as APIs do PySUS:\n  - " + "\n  - ".join(erros))


# =========================================================================== #
# 2. SIMULADOR DE CONTINGÊNCIA
# =========================================================================== #
def simular_sih(n: int = 40_000, seed: int = 20240501,
                uf_ibge: str = "31") -> pd.DataFrame:
    """Gera um AIH-RD sintético com o *mesmo schema e as mesmas codificações*
    do DATASUS (strings zero-padded, datas AAAAMMDD, `SEXO` em 1/3, etc.).

    Isto **não** é um substituto científico dos dados reais: serve para (i)
    testes automatizados, (ii) demonstração pública do repositório sem violar
    termos de uso, e (iii) verificação de que o pipeline recupera parâmetros
    conhecidos — um teste de recuperação de parâmetros ("parameter recovery"),
    prática padrão em validação de software estatístico.

    O mecanismo gerador é declarado explicitamente para permitir essa
    verificação:

    * Risco basal de óbito segue Weibull com forma < 1 (hazard decrescente:
      a letalidade intra-hospitalar concentra-se nos primeiros dias).
    * Log-HR verdadeiros de óbito: idade (+0,045/ano), UTI (+1,10),
      urgência (+0,35), comorbidade (+0,30), DPOC vs pneumonia (−0,25),
      asma vs pneumonia (−1,10), insuficiência respiratória (+0,45).
    * A alta hospitalar é um **evento competitivo** gerado por um processo de
      risco independente (Weibull), e observa-se o mínimo dos dois tempos —
      i.e., risco latente com censura por competição.
    * Transferências (≈3%) geram censura à direita genuína, permitindo que os
      pesos IPCW de Fine-Gray sejam não triviais.
    """
    rng = np.random.default_rng(seed)

    # ---------------------- Covariáveis latentes --------------------------- #
    # Idade: Gamma(6, 7) deslocada -> média ≈ 60 anos, cauda longa à direita,
    # compatível com o perfil etário das internações respiratórias adultas.
    idade = np.clip(rng.gamma(shape=6.0, scale=7.0, size=n) + 18, 18, 105)

    grupos = np.array(["Pneumonia", "DPOC", "Asma", "Insuf. respiratória", "Outras"])
    p_grupo = np.array([0.32, 0.30, 0.16, 0.13, 0.22])
    # A distribuição diagnóstica depende da idade (asma é mais jovem, DPOC mais
    # idoso): confundimento por indicação embutido de propósito, para que o
    # ajuste multivariável tenha o que corrigir.
    peso_idade = np.column_stack([
        np.ones(n),
        np.clip((idade - 40) / 30, 0, 3),
        np.clip((70 - idade) / 30, 0.05, 3),
        np.clip((idade - 50) / 30, 0.1, 3),
        np.ones(n),
    ])
    probs = peso_idade * p_grupo
    probs /= probs.sum(axis=1, keepdims=True)
    # Sorteio categórico vetorizado por inversão da acumulada (evita um laço
    # Python de n iterações — relevante para n na casa de 10^6).
    idx_grupo = (probs.cumsum(axis=1) < rng.uniform(size=(n, 1))).sum(axis=1)
    grupo = grupos[idx_grupo]

    sexo_f = rng.binomial(1, 0.47, n)
    nao_branca = rng.binomial(1, 0.58, n)
    urgencia = rng.binomial(1, np.where(np.isin(grupo, ["Asma", "DPOC"]), 0.93, 0.86))
    comorb = rng.binomial(1, np.clip(0.10 + 0.006 * (idade - 18), 0.05, 0.75))
    # UTI depende de gravidade latente -> confundidor forte do óbito
    logit_uti = -3.6 + 0.030 * idade + 0.85 * comorb + 0.55 * (grupo == "Insuf. respiratória")
    uti = rng.binomial(1, 1 / (1 + np.exp(-logit_uti)))
    mes = rng.integers(1, 13, n)
    inverno = np.isin(mes, [5, 6, 7, 8]).astype(int)   # sazonalidade respiratória

    # ------------------- Riscos de causa específica ------------------------ #
    lp_obito = (0.045 * (idade - 65) + 1.10 * uti + 0.35 * urgencia + 0.30 * comorb
                + 0.12 * inverno
                - 0.25 * (grupo == "DPOC") - 1.10 * (grupo == "Asma")
                + 0.45 * (grupo == "Insuf. respiratória") - 0.15 * (grupo == "Outras"))
    lp_alta = (-0.012 * (idade - 65) - 0.75 * uti - 0.30 * comorb
               + 0.55 * (grupo == "Asma") - 0.10 * (grupo == "DPOC"))

    # Weibull(forma, escala) por inversão: T = escala * (-log U)^(1/forma)
    forma_obito, forma_alta = 0.85, 1.25
    esc_obito = 320.0 * np.exp(-lp_obito / forma_obito)
    esc_alta = 6.5 * np.exp(-lp_alta / forma_alta)
    t_obito = esc_obito * (-np.log(rng.uniform(size=n))) ** (1 / forma_obito)
    t_alta = esc_alta * (-np.log(rng.uniform(size=n))) ** (1 / forma_alta)

    t_transf = rng.exponential(120.0, size=n)          # censura por transferência
    tempos = np.column_stack([t_obito, t_alta, t_transf])
    causa = tempos.argmin(axis=1)                      # 0 óbito, 1 alta, 2 transferência
    dias = np.floor(np.clip(tempos.min(axis=1), 0, 250)).astype(int)

    # -------------------- Codificação no padrão DATASUS -------------------- #
    cid_por_grupo = {
        "Pneumonia": ["J189", "J159", "J128", "J180", "J13", "J440"],
        "DPOC": ["J440", "J441", "J449", "J432", "J42"],
        "Asma": ["J450", "J451", "J458", "J46"],
        "Insuf. respiratória": ["J960", "J969", "J80", "J849"],
        "Outras": ["J690", "J208", "J38 3", "J939", "J90"],
    }
    diag = np.array([rng.choice(cid_por_grupo[g]) for g in grupo])
    # Ruído de classificação: ~1,5% dos códigos vêm com espaço/minúscula, como
    # de fato ocorre nos arquivos do DATASUS — o pré-processamento tem de limpar.
    ruim = rng.random(n) < 0.015
    diag = np.where(ruim, np.char.add(diag.astype(str), " "), diag)

    dt_inter = np.array([date(2023, int(m), 1) + timedelta(days=int(rng.integers(0, 27)))
                         for m in mes])
    dt_saida = np.array([d + timedelta(days=int(x)) for d, x in zip(dt_inter, dias)])
    nasc = np.array([d - timedelta(days=int(a * 365.25)) for d, a in zip(dt_inter, idade)])

    fmt = np.vectorize(lambda d: d.strftime("%Y%m%d"))

    # COBRANCA (motivo de saída): 11-19 alta; 31-33 transferência; 41-43 óbito
    cobranca = np.select(
        [causa == 0, causa == 1, causa == 2],
        [rng.choice(["41", "42", "43"], n, p=[0.86, 0.08, 0.06]),
         rng.choice(["11", "12", "14", "18"], n, p=[0.72, 0.14, 0.08, 0.06]),
         rng.choice(["31", "32"], n, p=[0.7, 0.3])],
        default="11",
    )

    df = pd.DataFrame({
        "N_AIH": np.char.add("31", rng.integers(10**10, 10**11, n).astype(str)),
        "IDENT": "1",
        "CNES": rng.choice([f"{c:07d}" for c in rng.integers(2000000, 2999999, 180)], n),
        "MUNIC_RES": rng.choice([uf_ibge + f"{c:04d}" for c in rng.integers(1000, 9999, 300)], n),
        "MUNIC_MOV": rng.choice([uf_ibge + f"{c:04d}" for c in rng.integers(1000, 9999, 90)], n),
        "NASC": fmt(nasc),
        "SEXO": np.where(sexo_f == 1, "3", "1"),
        "IDADE": np.round(idade).astype(int).astype(str),
        "COD_IDADE": "4",
        "RACA_COR": np.where(nao_branca == 1,
                             rng.choice(["02", "03", "04", "05"], n, p=[.16, .78, .04, .02]),
                             "01"),
        "DT_INTER": fmt(dt_inter),
        "DT_SAIDA": fmt(dt_saida),
        "DIAS_PERM": dias.astype(str),
        "MORTE": np.where(causa == 0, "1", "0"),
        "COBRANCA": cobranca,
        "DIAG_PRINC": diag,
        "DIAG_SECUN": np.where(comorb == 1,
                               rng.choice(["I500", "E149", "N180", "I10", "C349"], n),
                               rng.choice(["", "0000"], n, p=[.8, .2])),
        "CAR_INT": np.where(urgencia == 1, "02", "01"),
        "MARCA_UTI": np.where(uti == 1, rng.choice(["74", "75", "76", "81"], n), "00"),
        "UTI_MES_TO": np.where(uti == 1,
                               np.minimum(dias, rng.integers(1, 12, n)), 0).astype(str),
        "ESPEC": rng.choice(["01", "03", "05"], n, p=[.68, .22, .10]),
        "COMPLEX": np.where(uti == 1, "03", rng.choice(["02", "03"], n, p=[.85, .15])),
        "VAL_TOT": np.round(rng.lognormal(7.1 + 0.9 * uti, 0.55, n), 2).astype(str),
    })

    # Dados administrativos têm ausência não aleatória: RACA_COR é o campo mais
    # incompleto do SIH. Injetamos MAR (depende de idade/UTI) para exercitar a
    # imputação múltipla na análise de sensibilidade.
    p_missing = 1 / (1 + np.exp(-(-2.2 + 0.012 * (idade - 65) - 0.4 * uti)))
    df.loc[rng.random(n) < p_missing, "RACA_COR"] = "99"

    df.attrs["origem"] = "simulado"
    df.attrs["seed"] = seed
    return df


# =========================================================================== #
# 3. FACHADA PÚBLICA
# =========================================================================== #
def obter_coorte(cfg: AnalysisConfig, forcar_download: bool = False) -> pd.DataFrame:
    """Devolve a AIH-RD bruta, resolvendo cache → PySUS → simulação.

    Parameters
    ----------
    cfg : AnalysisConfig
    forcar_download : bool
        Ignora o cache em Parquet e vai à fonte.

    Notes
    -----
    O cache é nomeado por UF/anos/grupo. Se você alterar a janela do estudo,
    um novo arquivo é criado — o antigo permanece, preservando a auditabilidade
    de análises anteriores.
    """
    cfg.preparar_diretorios()
    nome = f"sih_{cfg.grupo_sih}_{cfg.uf}_{min(cfg.anos)}-{max(cfg.anos)}.parquet"
    cache = cfg.dir_dados / "raw" / nome

    if cache.exists() and not forcar_download:
        log.info("Lendo cache local: %s", cache)
        df = pd.read_parquet(cache)
        df.attrs.setdefault("origem", "cache")
        return df

    if cfg.modo_dados in ("auto", "pysus"):
        try:
            df = baixar_sih(cfg)
            faltantes = [c for c in COLUNAS_RD if c not in df.columns]
            if faltantes:
                warnings.warn(f"Colunas ausentes no arquivo do DATASUS: {faltantes}")
            df = df[[c for c in COLUNAS_RD if c in df.columns]].copy()
            df.attrs["origem"] = "datasus"
            df.to_parquet(cache, index=False)
            return df
        except Exception as exc:                       # noqa: BLE001
            if cfg.modo_dados == "pysus":
                raise
            log.warning("Download indisponível (%s). Usando coorte simulada.", exc)

    df = simular_sih(n=cfg.n_simulacao, seed=cfg.seed)
    df.to_parquet(cache.with_name(cache.stem + "_SIMULADO.parquet"), index=False)
    return df
