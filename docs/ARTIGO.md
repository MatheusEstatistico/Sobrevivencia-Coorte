# Sobrevida intra-hospitalar em internações por doenças do aparelho respiratório em Minas Gerais: uma coorte retrospectiva com análise de riscos competitivos

**Documentação metodológica em formato de manuscrito**
Versão 1.0 · Repositório: `Sobrevivencia-Coorte`

> **Aviso de escopo.** Este documento é a **documentação metodológica** do
> pipeline, redigida no formato de um manuscrito para explicitar cada decisão
> analítica. Os valores numéricos citados na seção Resultados foram obtidos com
> a **coorte sintética** distribuída com o repositório (`--modo simular`), cujos
> parâmetros geradores são conhecidos e servem para verificar se o código os
> recupera. Ao executar com `--modo pysus`, as mesmas tabelas e figuras são
> regeradas com os microdados reais do SIH/SUS, e os números mudam. Nenhum
> resultado aqui deve ser citado como estimativa epidemiológica.

---

## Resumo estruturado

**Objetivo.** Estimar e comparar o risco de óbito intra-hospitalar entre grupos
diagnósticos de doenças respiratórias (CID-10 J00–J99) em internações do SUS em
Minas Gerais, tratando a alta hospitalar como evento competitivo, e quantificar
o viés introduzido por sua desconsideração.

**Métodos.** Coorte retrospectiva construída a partir das Autorizações de
Internação Hospitalar (AIH) do Sistema de Informações Hospitalares (SIH/SUS),
obtidas via biblioteca `PySUS`. Incluídas internações de adultos (≥ 18 anos) com
diagnóstico principal do capítulo X da CID-10. O tempo de origem foi a data de
internação e o seguimento, truncado administrativamente em 30 dias. Os desfechos
foram tratados como estados mutuamente exclusivos: óbito intra-hospitalar
(evento de interesse), alta hospitalar (evento competitivo) e censura
(transferência ou permanência além de 30 dias). Estimaram-se curvas de
Kaplan-Meier estratificadas, com teste de log-rank global e par a par
(correção de Holm), e o tempo médio restrito de sobrevida (RMST). O risco
absoluto foi estimado pela incidência acumulada de Aalen-Johansen, com intervalos
de confiança por bootstrap. A modelagem multivariável empregou (i) regressão de
Cox para o risco de causa específica, com idade em spline cúbica restrita,
erros-padrão robustos agrupados por estabelecimento e verificação de riscos
proporcionais por resíduos de Schoenfeld, e (ii) o modelo de sub-distribuição de
Fine-Gray, implementado por ponderação IPCW do conjunto de risco. A discriminação
foi validada internamente por bootstrap (correção de otimismo de Harrell) e a
robustez a confundimento não medido, por E-values.

**Resultados (coorte sintética de verificação).** Entre 40.000 internações
elegíveis ocorreram 4.011 óbitos (10,0%). A incidência acumulada de óbito em 30
dias variou de 1,0% (asma) a 27,7% (insuficiência respiratória). Tratar a alta
como censura produziu risco estimado de 36,2% aos 30 dias — **3,45 vezes** a
incidência acumulada correta de 10,5%. No modelo de Cox ajustado, associaram-se
a maior risco o uso de UTI (HR 3,09; IC95% 2,90–3,30), a internação de urgência
(HR 1,51; IC95% 1,34–1,70) e a insuficiência respiratória em relação à pneumonia
(HR 1,39; IC95% 1,26–1,54). O sHR do uso de UTI no modelo de Fine-Gray (4,11)
excedeu o HR de causa específica, refletindo o efeito simultâneo da UTI em
retardar a alta. O C-index corrigido por otimismo foi 0,808.

**Conclusão.** Em coortes hospitalares, nas quais mais de 90% dos indivíduos
saem do estado de risco por alta, o Kaplan-Meier superestima grosseiramente o
risco absoluto de óbito. A apresentação conjunta de modelos de causa específica
e de sub-distribuição é necessária, pois respondem a perguntas distintas e podem
divergir sistematicamente.

**Palavras-chave.** Análise de sobrevida; riscos competitivos; Sistema de
Informações Hospitalares; doenças respiratórias; mortalidade hospitalar.

---

## 1. Introdução

As doenças do aparelho respiratório estão entre as principais causas de
internação e de óbito hospitalar no Brasil, com marcada sazonalidade e forte
gradiente etário. O SIH/SUS registra praticamente a totalidade das internações
financiadas pelo sistema público, o que o torna uma fonte de dados de cobertura
populacional para o estudo de desfechos hospitalares — com as limitações
próprias de um sistema concebido para faturamento, e não para pesquisa.

A literatura aplicada que usa esses dados frequentemente reporta letalidade
bruta (óbitos ÷ internações) ou curvas de Kaplan-Meier em que a alta hospitalar
é tratada como censura. As duas práticas são problemáticas. A letalidade bruta
ignora o tempo sob risco e é sensível a diferenças de permanência entre grupos.
O Kaplan-Meier com alta censurada viola o pressuposto de censura não informativa:
um paciente que recebeu alta não permanece sob risco de óbito *intra-hospitalar*,
de modo que o estimador responde à pergunta contrafactual "qual seria o risco de
óbito se a alta pudesse ser indefinidamente adiada" — uma quantidade sem
correspondência clínica.

Este trabalho tem três objetivos:

1. **Substantivo.** Estimar e comparar o risco de óbito intra-hospitalar entre
   grupos diagnósticos respiratórios, ajustado por características do paciente e
   da internação.
2. **Metodológico.** Quantificar empiricamente a magnitude do viés do
   Kaplan-Meier nesse contexto e contrastar as inferências dos modelos de causa
   específica e de sub-distribuição.
3. **De engenharia.** Disponibilizar um pipeline reprodutível, testado e
   documentado, que vá do download dos microdados às figuras finais.

---

## 2. Métodos

### 2.1 Desenho e fonte de dados

Coorte retrospectiva de base hospitalar. Os dados provêm do arquivo **RD (AIH
Reduzida)** do SIH/SUS, obtido do FTP do DATASUS por meio da biblioteca
`PySUS`. Cada registro corresponde a uma AIH encerrada, contendo diagnóstico
principal e secundário (CID-10), datas de internação e saída, dias de
permanência, indicador de óbito, motivo de saída, uso de UTI, caráter da
internação e identificação do estabelecimento (CNES).

O módulo de aquisição implementa três camadas: consulta ao DATASUS, cache local
em Parquet e um gerador sintético de contingência. O cache é essencial para
reprodutibilidade, porque o DATASUS **retifica arquivos retroativamente** — uma
competência baixada hoje pode diferir da mesma competência baixada em três
meses. Sem materialização datada, nenhuma análise sobre esses dados é
estritamente reproduzível.

### 2.2 Critérios de elegibilidade

Foram incluídas AIH que atendessem simultaneamente a:

| # | Critério | Justificativa |
|---|----------|---------------|
| 1 | AIH do tipo normal (`IDENT` = 1) | AIH de continuação (`IDENT` = 5) prorrogam uma internação já contabilizada; incluí-las duplicaria indivíduos e violaria a independência das observações |
| 2 | Diagnóstico principal J00–J99 | Capítulo X da CID-10 — doenças do aparelho respiratório |
| 3 | Idade entre 18 e 110 anos | A mortalidade pediátrica por causa respiratória tem determinantes e hazard distintos; o limite superior é controle de qualidade |
| 4 | Permanência válida (0–365 dias) | Exclui inconsistências de datas (saída anterior à entrada) e outliers implausíveis |

O fluxograma de elegibilidade (Figura 1 do padrão STROBE) é gerado
automaticamente em `outputs/tabelas/fluxograma.txt`, com o número de registros
excluídos em cada etapa.

### 2.3 Definição do tempo e dos desfechos

**Origem do tempo (t = 0):** data de internação. Não há truncamento à esquerda,
pois a coorte é definida no momento da admissão — todos os indivíduos entram sob
risco simultaneamente na escala de tempo do estudo.

**Escala de tempo:** dias de permanência. Internações com alta ou óbito no mesmo
dia (`DIAS_PERM` = 0) receberam tempo 0,5, equivalente a assumir que o evento
ocorreu no ponto médio do único intervalo observável. Tempo zero é inadmissível
em modelos de sobrevida.

**Estados de saída** (mutuamente exclusivos, derivados de `MORTE` e `COBRANCA`):

| Código | Estado | Regra |
|--------|--------|-------|
| 1 | Óbito intra-hospitalar | `MORTE` = 1 **ou** `COBRANCA` ∈ [41, 43] |
| 2 | Alta hospitalar (vivo) | `COBRANCA` ∈ [11, 19] |
| 0 | Censura | `COBRANCA` ∈ [31, 33] (transferência) ou permanência > 30 dias |

O óbito domina a hierarquia porque o registro de óbito no SIH vincula-se à
emissão da Declaração de Óbito, sendo o campo mais confiável do sistema.

**Truncamento administrativo:** seguimento limitado a τ = 30 dias. Além de
alinhar-se ao indicador padrão de mortalidade hospitalar em 30 dias, o
truncamento estabiliza a cauda das curvas, onde o número sob risco é pequeno e a
variância do estimador explode. Pacientes ainda internados em t = 30 foram
censurados administrativamente.

### 2.4 Variáveis

**Exposição principal:** grupo diagnóstico, definido *a priori* por homogeneidade
fisiopatológica e prognóstica:

| Grupo | Códigos CID-10 |
|-------|----------------|
| Pneumonia (referência) | J12–J18 |
| DPOC | J40–J44 |
| Asma | J45–J46 |
| Insuficiência respiratória | J80–J84, J96 |
| Outras respiratórias | demais J |

A pneumonia foi escolhida como referência por ser o grupo mais frequente e o
comparador clinicamente mais interpretável.

**Covariáveis de ajuste** (selecionadas *a priori* por conhecimento causal, não
por significância estatística): idade (spline cúbica restrita, 3 g.l.), sexo,
raça/cor dicotomizada, uso de UTI, caráter de urgência, presença de diagnóstico
secundário (proxy de comorbidade) e internação no inverno (maio–agosto, período
de circulação de vírus respiratórios no Sudeste).

**Estrutura causal presumida.** O conjunto de ajuste segue o critério de
*backdoor*: idade, sexo e raça/cor são causas comuns do grupo diagnóstico e do
óbito. O uso de UTI merece nota — trata-se de um **mediador parcial** do efeito
do diagnóstico sobre o óbito e, simultaneamente, de um marcador de gravidade não
observada. Ajustar por ele produz um efeito direto controlado, não o efeito
total, e pode introduzir viés de colisor se houver causa comum não medida entre
UTI e óbito (gravidade clínica). Reportamos ambos os modelos, com e sem UTI, e
discutimos a interpretação na seção 4.3. Esta é uma limitação de desenho, não um
detalhe de implementação.

### 2.5 Análise estatística

#### 2.5.1 Descrição basal

Tabela 1 com **diferenças médias padronizadas (SMD)** em vez de p-valores.
Testes de hipótese para desequilíbrio basal são desencorajados em estudos
observacionais: com n na casa das dezenas de milhares, diferenças irrelevantes
atingem p < 0,001, e a coorte **é** a população de interesse — não uma amostra
dela. A SMD independe do n; adota-se 0,10 como limiar de desequilíbrio não
desprezível. Variáveis contínuas são resumidas por mediana [IQR] quando
assimétricas (|g₁| > 1) e por média (DP) caso contrário, decisão tomada por
variável e registrada na própria linha da tabela.

Reporta-se também a **densidade de incidência** de óbito por 1.000 pessoas-dia,
com IC de Poisson (método de Byar), que complementa as medidas de risco quando a
permanência difere entre grupos.

#### 2.5.2 Estimação não paramétrica

Curvas de Kaplan-Meier estratificadas por grupo diagnóstico, com bandas de
confiança e tabela de números sob risco. Comparação por log-rank global
(*k* amostras) e par a par, com correção de **Holm-Bonferroni**: com 5 grupos há
10 comparações, e sem correção a probabilidade de ao menos um falso positivo a
5% seria de aproximadamente 40%. Holm foi preferido a Benjamini-Hochberg porque
os pares compartilham dados e são correlacionados, o que viola o pressuposto de
independência do BH em sua forma simples.

**RMST.** O tempo médio restrito de sobrevida, RMST(τ) = ∫₀^τ S(u)du, é
reportado como medida de efeito complementar. A razão de riscos só tem
interpretação simples sob proporcionalidade; quando esta falha, o HR estimado é
uma média ponderada dos hazards instantâneos com pesos que dependem da
distribuição de censura da própria amostra. O RMST não sofre disso e se
interpreta em unidade de tempo. Sua variância é obtida pela fórmula de
Klein & Moeschberger, análoga a Greenwood para a área sob a curva.

#### 2.5.3 Riscos competitivos

O risco absoluto é estimado pelo **estimador de Aalen-Johansen**:

$$\hat F_k(t) = \sum_{t_j \le t} \hat S(t_{j-1}) \frac{d_{kj}}{n_j}$$

em que $\hat S$ é o Kaplan-Meier de "qualquer saída do hospital". O fator
$\hat S(t_{j-1})$ é exatamente o que falta ao KM ingênuo: só pode morrer no
instante *t* quem ainda está internado em *t*. A implementação é própria, sem
*jitter* de empates — inaceitável em permanência medida em dias inteiros, em que
os empates são a regra. Os IC95% são percentílicos por bootstrap, o que respeita
os limites [0, 1] sem transformação *ad hoc*.

A igualdade das CIFs entre grupos é testada por **permutação** dos rótulos de
grupo (mantendo fixos os pares tempo–status, o que preserva as estruturas de
censura e de competição), com estatística igual à soma dos desvios quadráticos
integrados das CIFs de grupo em relação à CIF combinada. O teste *k*-amostral de
Gray (1988) não está implementado nas bibliotecas Python de uso corrente; o teste
de permutação é sua alternativa exata-condicional. O p-valor usa a correção
(1 + #{T_perm ≥ T_obs}) / (1 + B), válida para qualquer B e que nunca retorna
zero exato.

#### 2.5.4 Modelagem multivariável

**Modelo de causa específica.** Cox com aproximação de **Efron** para empates
(sensivelmente menos enviesada que Breslow quando os empates são numerosos, como
aqui). Decisões:

- *Idade em spline cúbica restrita (3 g.l., nós nos percentis 5/35/65/95).*
  Categorizar covariáveis contínuas descarta informação e cria confundimento
  residual. A restrição de linearidade além dos nós externos evita comportamento
  errático nas caudas. Como a primeira coluna da base é o próprio *x*, testar a
  nulidade das demais é um teste formal de não linearidade.
- *Erros-padrão robustos agrupados por CNES.* Pacientes do mesmo estabelecimento
  compartilham protocolos e perfil de gravidade; ignorar a correlação
  intraclasse subestima a variância.
- *Verificação de riscos proporcionais* por resíduos de Schoenfeld escalonados
  (Grambsch & Therneau). Covariáveis que violam o pressuposto migram para o termo
  de estratificação, que concede risco basal próprio a cada estrato ao custo de
  não estimar um HR para a variável.

**Modelo de sub-distribuição (Fine-Gray).** Implementado do zero pela
equivalência de **Geskus (2011)**: o modelo de sub-distribuição é um Cox
ponderado no qual os indivíduos que sofrem o evento competitivo **permanecem no
conjunto de risco** após o evento, com peso decrescente

$$w_i(t) = \frac{\hat G(t)}{\hat G(T_i)}, \quad t > T_i$$

em que $\hat G$ é o Kaplan-Meier da distribuição de censura (IPCW). O
procedimento replica `survival::finegray()` do R, e o ajuste torna-se um Cox com
intervalos (start, stop] e pesos. Como $\hat G$ só decresce nos tempos de
censura, basta uma linha por tempo de censura posterior a $T_i$ — o custo é
O(n · #tempos de censura), e não O(n²).

*Limitação conhecida.* A variância de Fine-Gray corrige o fato de os pesos serem
estimados; o `lifelines` não implementa variância robusta em modelos com
covariáveis dependentes do tempo (`NotImplementedError`). O padrão adotado é a
variância do modelo, com IC por bootstrap disponível via `n_boot_ic`. Com censura
leve (~5% nesta coorte) a diferença é pequena, mas o usuário deve ativar o
bootstrap em cenários de censura pesada.

**Interpretação.** Os dois modelos respondem a perguntas diferentes e não são
intercambiáveis:

| Modelo | Estima | Serve para |
|--------|--------|-----------|
| Causa específica | h₁(t) entre os que ainda estão internados | Etiologia, mecanismo |
| Fine-Gray | efeito sobre F₁(t) diretamente | Prognóstico, risco absoluto, decisão |

Uma covariável pode elevar o hazard de causa específica sem alterar a CIF, se
também acelerar o evento competitivo.

#### 2.5.5 Validação e análises de sensibilidade

- **Discriminação:** C-index corrigido pelo **otimismo de Harrell** por bootstrap
  (B = 200). O bootstrap usa 100% dos dados para desenvolvimento e tem menor
  variância que uma divisão treino/teste única, sendo a validação interna
  preferencial segundo o TRIPOD.
- **E-values** (VanderWeele & Ding) para a exposição principal: força mínima de
  associação que um confundidor não medido precisaria ter, com a exposição e com
  o desfecho, para explicar integralmente o efeito observado.
- **Modelo estratificado por UTI** como sensibilidade à não proporcionalidade.
- **Dados ausentes:** relatório de ausência com teste de associação ao desfecho.
  Se a probabilidade de ausência difere entre quem morreu e quem não morreu, o
  mecanismo não é MCAR e a análise de casos completos pode ser enviesada.

#### 2.5.6 Reprodutibilidade

Todos os parâmetros de desenho residem em uma dataclass congelada
(`AnalysisConfig`), versionada no Git: qualquer alteração de critério aparece no
`git diff`, funcionando como plano de análise estatística executável. Sementes
fixas; metadados de execução (versão do Python, plataforma, origem dos dados,
duração) gravados em `outputs/metadados_execucao.json`.

---

## 3. Resultados

> Números obtidos com a coorte sintética de verificação (n = 40.000, semente
> 20240501, execução completa em 180 s). Servem para demonstrar o funcionamento
> do pipeline e a recuperação dos parâmetros geradores conhecidos.

### 3.1 Coorte

Das 40.000 AIH simuladas, todas foram elegíveis (a simulação não gera registros
inválidos; com dados reais o fluxograma registra as exclusões). Ocorreram
**4.011 óbitos (10,0%)**, 33.681 altas (84,2%) e 2.308 censuras (5,8%), estas
por transferência ou permanência além de 30 dias. A idade média foi de 59,8
anos e a permanência mediana, de 4 dias. Raça/cor esteve ausente em 8,7% dos
registros; a proporção de ausência foi semelhante entre óbitos (9,0%) e não
óbitos (8,7%), compatível com MCAR nesta coorte (Tabela S1). O modelo ajustado,
por usar casos completos, foi estimado com 36.522 internações e 3.651 óbitos.

### 3.2 Verificação de recuperação de parâmetros

O teste decisivo de um pipeline estatístico é recuperar parâmetros conhecidos.
Comparando os coeficientes verdadeiros do gerador com as estimativas do modelo
de Cox de causa específica ajustado:

| Covariável | HR verdadeiro | HR estimado (IC95%) |
|------------|---------------|---------------------|
| Uso de UTI | 3,00 | 3,09 (2,90–3,30) |
| Internação de urgência | 1,42 | 1,51 (1,34–1,70) |
| Comorbidade registrada | 1,35 | 1,37 (1,28–1,47) |
| Internação no inverno | 1,13 | 1,13 (1,05–1,21) |
| DPOC vs pneumonia | 0,78 | 0,76 (0,70–0,82) |
| Asma vs pneumonia | 0,33 | 0,30 (0,21–0,43) |
| Insuf. respiratória vs pneumonia | 1,57 | 1,39 (1,26–1,54) |
| Sexo feminino (sem efeito) | 1,00 | 1,01 (0,94–1,07) |
| Raça/cor não branca (sem efeito) | 1,00 | 1,07 (1,00–1,14) |

A recuperação é boa em magnitude e direção, e as duas covariáveis sem efeito no
gerador ficam junto à nulidade. O desvio na insuficiência respiratória (1,39
estimado contra 1,57 verdadeiro) é esperado e informativo: com forma de Weibull
0,85 (hazard decrescente) e escalas muito distintas entre grupos, o mecanismo
gerador **não** satisfaz riscos proporcionais, e o HR estimado é a média
ponderada discutida na seção 2.5.2 — precisamente o motivo pelo qual o RMST é
reportado em paralelo.

### 3.3 Viés do Kaplan-Meier (Figura 3)

Aos 30 dias, o risco de óbito estimado por 1 − KM foi de **36,2%**, contra
**10,5%** pela incidência acumulada de Aalen-Johansen: uma superestimação de
**3,45 vezes**. A razão entre os estimadores cresce monotonicamente com o
tempo, porque o viés é proporcional ao acúmulo do evento competitivo — e aos 30
dias mais de 84% da coorte já recebeu alta. Em coortes hospitalares esse é o
caso extremo em que o erro deixa de ser sutil e passa a inverter a ordem de
grandeza da conclusão.

### 3.4 Incidência acumulada por grupo (Figura 4, Tabela 5)

| Grupo | CIF de óbito em 30 dias | IC95% (bootstrap) | RMST (dias) |
|-------|------------------------|-------------------|-------------|
| Asma | 0,98% | 0,67–1,28% | 27,9 |
| Outras respiratórias | 8,22% | 7,68–8,76% | 24,9 |
| Pneumonia | 9,63% | 9,11–10,09% | 24,3 |
| DPOC | 13,56% | 12,95–14,26% | 23,2 |
| Insuficiência respiratória | 27,66% | 25,62–29,70% | 17,7 |

Log-rank global: χ² correspondente a p < 10⁻¹⁵⁰. Teste de permutação das CIFs:
p = 0,002 (B = 499), o menor valor distinguível com esse número de réplicas.

O RMST torna o achado tangível: um paciente internado por insuficiência
respiratória vive, em média, **6,6 dias a menos** dentro da janela de 30 dias do
que um internado por pneumonia — uma medida em unidade de tempo, que permanece
interpretável mesmo com os riscos não proporcionais detectados.

### 3.5 Causa específica vs Fine-Gray (Tabela 8)

A expansão de Fine-Gray gerou 518.993 linhas a partir de 25.000 internações
(fator 20,8×; peso IPCW médio 0,904). Duas divergências relevantes:

| Covariável | HR causa-específica | sHR (Fine-Gray) | Leitura |
|------------|--------------------|-----------------|---------|
| Uso de UTI | 3,09 | 4,11 | A UTI eleva o hazard de óbito **e** retarda a alta; o modelo de sub-distribuição soma os dois canais, e o efeito sobre o risco absoluto supera o efeito sobre o hazard instantâneo |
| Asma vs pneumonia | 0,30 | 0,24 | A asma tem alta rápida, o que retira o paciente do risco antes que o óbito se materialize; o efeito protetor sobre a CIF é maior que sobre o hazard |

As demais covariáveis foram concordantes (diferença ≤ 0,20 na escala log). Ou
seja: a divergência não é ruído — ela aparece exatamente onde a covariável
também afeta o evento competitivo, que é o comportamento previsto pela teoria.
Reportar apenas um dos modelos daria uma leitura incompleta, e a escolha de qual
enfatizar deve seguir a pergunta (etiológica ou prognóstica), não a conveniência.

### 3.6 Discriminação e sensibilidade

C-index aparente 0,808; otimismo por bootstrap (B = 200) de 0,0007; **C-index
corrigido 0,808**. O otimismo desprezível é coerente com a razão
eventos-por-variável muito elevada (4.011 óbitos para 13 parâmetros, EPV ≈ 308),
bem acima do mínimo convencional de 10.

Os E-values da exposição principal indicam robustez heterogênea: para a asma
versus pneumonia, um confundidor não medido precisaria estar associado a ambos
com RR ≥ 6,1 (≥ 4,1 para deslocar o limite do IC até a nulidade) — implausível.
Já para "outras respiratórias versus pneumonia" o E-value é 1,66 (limite do IC
1,38), de modo que um confundidor moderado bastaria para explicar o achado.

## 4. Discussão

### 4.1 Achados principais

Três resultados se destacam. Primeiro, a magnitude do viés do Kaplan-Meier em
coortes hospitalares é de várias vezes, não de alguns pontos percentuais — o que
torna a escolha do estimador uma questão de validade, não de refinamento.
Segundo, os modelos de causa específica e de sub-distribuição divergem de forma
sistemática e previsível para covariáveis que afetam ambos os desfechos, como o
uso de UTI. Terceiro, o gradiente de risco entre grupos diagnósticos permanece
substancial após ajuste, com a insuficiência respiratória concentrando o risco.

### 4.2 Limitações da fonte de dados

O SIH/SUS é um sistema de **faturamento**, não um prontuário. As limitações
decorrentes são estruturais e não podem ser contornadas por técnica estatística:

1. **Unidade de análise é a internação, não o paciente.** Sem identificador
   individual nos microdados públicos, reinternações do mesmo indivíduo entram
   como observações independentes. Isso subestima a variância e impede analisar
   trajetórias. O agrupamento por CNES atenua parte da correlação, mas não a de
   origem individual.
2. **Um único diagnóstico secundário.** A carga de comorbidade é
   sistematicamente subestimada, e índices como Charlson não são calculáveis com
   fidelidade. A variável "comorbidade registrada" é um proxy grosseiro.
3. **Seguimento limitado à internação.** Óbitos após a alta não são observados —
   o desfecho é mortalidade *intra-hospitalar*, não mortalidade em 30 dias. Uma
   política que reduza a mortalidade hospitalar antecipando altas melhoraria o
   indicador sem melhorar a sobrevida. O relacionamento probabilístico com o SIM
   (Sistema de Informações sobre Mortalidade) resolveria isso, mas não é viável
   com os microdados públicos, que não trazem identificadores nominais.
4. **Transferências como censura independente.** Pacientes transferidos são
   censurados, mas a transferência é plausivelmente informativa: transfere-se o
   paciente grave que a unidade não consegue manejar. Isso enviesa a CIF de óbito
   para baixo. Uma análise de sensibilidade tratando transferência como óbito
   fornece o limite superior do viés.
5. **Qualidade do registro varia entre estabelecimentos.** O campo raça/cor é o
   mais incompleto, e a ausência não é aleatória.
6. **Incentivos de codificação.** O diagnóstico registrado pode responder a
   incentivos de faturamento, o que introduz erro de classificação da exposição —
   provavelmente não diferencial em relação ao óbito, atenuando as estimativas em
   direção à nulidade.

### 4.3 Sobre o ajuste por uso de UTI

Vale insistir no ponto levantado em 2.4. O uso de UTI está no caminho causal
entre gravidade e óbito, e é ele próprio indicado pela gravidade. Ajustar por ele
responde "qual o efeito do diagnóstico sobre o óbito, fixado o nível de cuidado
intensivo" — um efeito direto controlado, útil para comparar desempenho
assistencial, mas **não** o efeito total do diagnóstico. Se houver gravidade não
medida causando tanto a admissão em UTI quanto o óbito, o ajuste abre um caminho
de colisor e pode enviesar em direção imprevisível. Por isso o pipeline reporta
também o modelo sem UTI, e nenhuma das estimativas deve ser lida como efeito
causal total sem discussão explícita da estrutura assumida.

### 4.4 Implicações metodológicas

Para quem analisa dados hospitalares do SUS, a recomendação prática é direta:
(i) sempre reportar a incidência acumulada, nunca 1 − KM, para risco absoluto de
óbito intra-hospitalar; (ii) declarar explicitamente qual pergunta — etiológica
ou prognóstica — motiva a escolha entre causa específica e sub-distribuição, e
preferencialmente reportar ambos; (iii) verificar riscos proporcionais e, quando
violados, complementar o HR com RMST, que permanece interpretável.

---

## 5. Conformidade com o STROBE

| Item | Onde |
|------|------|
| 1 Título e resumo | Título; Resumo estruturado |
| 2 Contexto/racional | §1 |
| 3 Objetivos | §1, itens 1–3 |
| 4 Desenho | §2.1 |
| 5 Contexto (local, período) | §2.1; `AnalysisConfig` |
| 6 Participantes | §2.2; fluxograma automático |
| 7 Variáveis | §2.4 |
| 8 Fontes de dados/mensuração | §2.1, §2.3 |
| 9 Vieses | §4.2, §4.3 |
| 10 Tamanho amostral | Coorte censitária; EPV em §3.6 |
| 11 Variáveis quantitativas | §2.4 (spline, sem categorização) |
| 12 Métodos estatísticos | §2.5 |
| 13 Participantes (fluxo) | `outputs/tabelas/fluxograma.txt` |
| 14 Dados descritivos | Tabela 1; Tabela S1 (ausências) |
| 15 Dados de desfecho | Tabelas 2, 3, 5 |
| 16 Resultados principais | Tabelas 6b, 7, 8; Figuras 6, 9 |
| 17 Outras análises | Tabelas S6–S8 (estratificado, bootstrap, E-values) |
| 18 Resultados-chave | §3, §4.1 |
| 19 Limitações | §4.2, §4.3 |
| 20 Interpretação | §4.4 |
| 21 Generalização | §4.2 |
| 22 Financiamento | Não aplicável (projeto de portfólio) |

---

## 6. Considerações éticas

Os microdados do SIH/SUS são de domínio público, anonimizados e agregados no
nível da internação, dispensando apreciação por Comitê de Ética em Pesquisa nos
termos da Resolução CNS nº 510/2016, Art. 1º, parágrafo único, incisos III e V.
Ainda assim, análises em recortes geográficos muito finos (município de pequeno
porte × faixa etária × diagnóstico raro) podem permitir reidentificação; o
pipeline não produz saídas nesse nível de desagregação.

---

## 7. Referências

1. Cox DR. Regression models and life-tables. *J R Stat Soc B*. 1972;34(2):187–220.
2. Kaplan EL, Meier P. Nonparametric estimation from incomplete observations. *J Am Stat Assoc*. 1958;53(282):457–481.
3. Aalen OO, Johansen S. An empirical transition matrix for non-homogeneous Markov chains based on censored observations. *Scand J Stat*. 1978;5(3):141–150.
4. Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. *J Am Stat Assoc*. 1999;94(446):496–509.
5. Gray RJ. A class of K-sample tests for comparing the cumulative incidence of a competing risk. *Ann Stat*. 1988;16(3):1141–1154.
6. Geskus RB. Cause-specific cumulative incidence estimation and the Fine and Gray model under both left truncation and right censoring. *Biometrics*. 2011;67(1):39–49.
7. Austin PC, Lee DS, Fine JP. Introduction to the analysis of survival data in the presence of competing risks. *Circulation*. 2016;133(6):601–609.
8. Latouche A, Allignol A, Beyersmann J, Labopin M, Fine JP. A competing risks analysis should report results on all cause-specific hazards and cumulative incidence functions. *J Clin Epidemiol*. 2013;66(6):648–653.
9. Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika*. 1994;81(3):515–526.
10. Harrell FE Jr. *Regression Modeling Strategies*. 2ª ed. Springer; 2015.
11. Harrell FE Jr, Lee KL, Mark DB. Multivariable prognostic models. *Stat Med*. 1996;15(4):361–387.
12. Royston P, Altman DG, Sauerbrei W. Dichotomizing continuous predictors in multiple regression: a bad idea. *Stat Med*. 2006;25(1):127–141.
13. VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med*. 2017;167(4):268–274.
14. Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. *Stat Med*. 2009;28(25):3083–3107.
15. Klein JP, Moeschberger ML. *Survival Analysis: Techniques for Censored and Truncated Data*. 2ª ed. Springer; 2003.
16. Hernán MA. The hazards of hazard ratios. *Epidemiology*. 2010;21(1):13–15.
17. Uno H, Claggett B, Tian L, et al. Moving beyond the hazard ratio in quantifying the between-group difference in survival analysis. *J Clin Oncol*. 2014;32(22):2380–2385.
18. von Elm E, Altman DG, Egger M, et al. The STROBE Statement. *Lancet*. 2007;370(9596):1453–1457.
19. Collins GS, Reitsma JB, Altman DG, Moons KGM. TRIPOD Statement. *Ann Intern Med*. 2015;162(1):55–63.
20. Coelho FC, et al. PySUS: a Python package for the Brazilian Unified Health System open data. (documentação: https://pysus.readthedocs.io)
