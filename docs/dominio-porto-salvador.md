# Domínio: Portos — Estudo de Caso Porto de Salvador (BA)

> Documento de referência para o projeto de **Image Captioning** aplicado ao domínio portuário.
> Objetivo: mapear conceitos, infraestrutura, operações e vocabulário visual necessários para
> entender e legendar corretamente imagens de um porto, usando o **Porto de Salvador** como
> estudo de caso principal.

---

## 1. O que é um porto

Um porto é uma infraestrutura de transição entre o modal marítimo/fluvial e os modais terrestres
(rodoviário, ferroviário, dutoviário), projetada para permitir a atracação segura de embarcações
e a movimentação eficiente de cargas e/ou passageiros entre navio e terra.

Um porto "funcionando" depende de três grandes camadas:

1. **Infraestrutura física** (o que existe: cais, canais, armazéns, guindastes).
2. **Operação e logística** (o que acontece: atracação, carga/descarga, armazenagem, transporte).
3. **Governança e regulação** (quem administra, fiscaliza e autoriza).

---

## 2. Infraestrutura física necessária

### 2.1 Infraestrutura de acesso aquaviário
- **Canal de acesso / bacia de evolução**: rota dragada com profundidade e largura suficientes
  para os navios manobrarem até o cais.
- **Dragagem**: manutenção da profundidade do canal e do berço (calado).
- **Sinalização náutica**: boias, faróis, balizas — orientam a navegação segura.
- **Quebra-mar / molhe**: estrutura que protege a bacia portuária de ondas e correntes.

### 2.2 Infraestrutura de atracação
- **Cais (berço de atracação)**: estrutura vertical de concreto/aço onde o navio encosta.
- **Defensas (fenders)**: amortecedores de borracha entre casco e cais.
- **Cabeços de amarração (bollards/cabeços)**: pontos de fixação dos cabos do navio.
- **Duque d'Alba**: estruturas isoladas de amarração/apoio fora do cais principal.

### 2.3 Equipamentos de movimentação de carga
- **Guindastes de cais (portêineres / STS cranes)**: grandes pórticos sobre trilhos, usados para
  carga/descarga de contêineres direto do navio.
- **Transtêineres (RTG — rubber-tyred gantry)**: movimentam contêineres no pátio.
- **Reach stackers e empilhadeiras**: manuseiam contêineres e cargas em pátio/armazém.
- **Guindastes móveis e guindastes de bordo (do próprio navio)**.
- **Correias transportadoras e tremonhas**: para granéis sólidos (minério, grãos, açúcar).
- **Braços de carregamento (loading arms) e mangotes**: para granéis líquidos (combustíveis,
  óleos) e gases.
- **Silos e tanques**: armazenamento de granel sólido e líquido.

### 2.4 Áreas de armazenagem
- **Pátio de contêineres**: área pavimentada para empilhamento de contêineres (vazios e cheios).
- **Armazéns/entrepostos**: cobertos, para carga geral e carga sensível a intempéries.
- **Terminal de granéis**: pátios abertos (minério, carvão) ou silos (grãos).
- **Zona alfandegada**: área sob controle aduaneiro antes da liberação da carga.

### 2.5 Infraestrutura terrestre de conexão
- **Vias de acesso rodoviário** e pátios de espera de caminhões.
- **Ramal ferroviário** conectando o porto à malha nacional (quando existir).
- **Retroárea logística**: centros de distribuição próximos ao porto.

### 2.6 Infraestrutura de apoio à navegação e segurança
- **Praticagem**: pilotos práticos que assumem o comando temporário do navio nas manobras.
- **Rebocadores**: embarcações que auxiliam a manobra de navios de grande porte.
- **Lanchas de apoio e embarcações de fiscalização** (Capitania dos Portos, Receita Federal).
- **Torre de controle / VTS (Vessel Traffic Service)**: monitora o tráfego de embarcações.
- **Sistemas de combate a incêndio e resposta a vazamentos**.

---

## 3. Operação portuária (o que acontece no dia a dia)

1. **Programação de atracação**: fila de navios, previsão de chegada (ETA) e janela de berço.
2. **Praticagem e rebocagem**: condução do navio até o berço.
3. **Amarração**: fixação do navio ao cais.
4. **Carga/descarga**: operação com guindastes/correias, conforme o tipo de carga.
5. **Conferência e vistoria**: verificação de quantidade/qualidade da carga, inspeção aduaneira e
   fitossanitária (Receita Federal, Vigiagro, Anvisa).
6. **Armazenagem temporária**: no pátio, armazém ou silo, até a retirada.
7. **Desembaraço aduaneiro**: liberação documental/fiscal da carga (importação/exportação).
8. **Transporte terrestre**: retirada por caminhão ou trem até o destino final.

### Atores envolvidos
- **Autoridade Portuária** (administração do porto).
- **Operadores portuários / arrendatários** (empresas privadas que operam terminais).
- **Agentes marítimos** (representam os armadores/navios).
- **Praticagem** (pilotos).
- **Órgãos anuentes**: Receita Federal, Capitania dos Portos (Marinha), Vigiagro/MAPA, Anvisa,
  Polícia Federal.
- **Sindicatos de trabalhadores portuários avulsos (OGMO)**.

---

## 4. Estudo de caso: Porto de Salvador (BA)

### 4.1 Localização e contexto geográfico
O Porto de Salvador está situado na **Baía de Todos os Santos**, no centro histórico da cidade de
Salvador, próximo à Cidade Baixa e ao Comércio. É um porto urbano, o que o diferencia de portos
mais afastados dos centros urbanos — nas imagens, é comum aparecer ao fundo o skyline da cidade,
o Elevador Lacerda e a orla.

### 4.2 Administração
- Administrado pela **CODEBA — Companhia das Docas do Estado da Bahia**, empresa pública federal
  vinculada ao Ministério de Portos e Aeroportos.
- A CODEBA também administra os portos de **Ilhéus** e **Aratu** na Bahia.
- Regulação e fiscalização do tráfego aquaviário: **Capitania dos Portos da Bahia** (Marinha do
  Brasil).

### 4.3 Terminais e arrendamentos em operação
- **Tecon Salvador** (operado pela **Wilson Sons**): terminal de contêineres e carga geral.
  - Área de aprox. 163.200 m², com 800 m de cais contínuo, capacidade para navios de até 366 m.
  - Capacidade de movimentação: ~535.000 TEU/ano (com plano de expansão para ~924.000 TEU/ano).
  - Equipamentos: portêineres (STS cranes), transtêineres (RTG), empilhadeiras reach stacker.
  - Reconhecido pelo Banco Mundial como um dos melhores terminais de contêineres do mundo na
    categoria até 500 mil TEUs.
- **Intermarítima Terminais Ltda**: terminal de contêineres, ~20.000 m², capacidade estática de
  3.000 TEU, uso de reach stackers.
- **FERBASA**: armazenagem de ferro-ligas para exportação (granel sólido/carga unitizada).
- **Corcovado do Nordeste**: armazenagem de granito para exportação (blocos de rocha ornamental).
- Recebe também **navios de cruzeiro** (terminal de passageiros).

### 4.4 Perfil de cargas
O Porto de Salvador é fortemente **exportador**, com destaque para:
- Contêineres (carga geral conteinerizada).
- Trigo (importação de granel sólido).
- Celulose.
- Ferro-ligas.
- Granito e rochas ornamentais.
- Frutas (o porto/estado tem relevância nas exportações de frutas do Brasil).
- Passageiros de cruzeiro (sazonal, temporada de verão).

### 4.5 Elementos visuais característicos do Porto de Salvador
Para o dataset de image captioning, imagens do Porto de Salvador tendem a conter:
- Navios porta-contêineres atracados no Tecon, com portêineres (guindastes de pórtico) ao lado.
- Pátios de contêineres coloridos (padrão ISO 20/40 pés) organizados em blocos.
- Navios graneleiros e correias transportadoras (granito, ferro-ligas).
- Navios de cruzeiro no terminal de passageiros, com vista para o Elevador Lacerda/Mercado
  Modelo ao fundo.
- Rebocadores manobrando na Baía de Todos os Santos.
- Guindastes móveis e empilhadeiras (reach stacker) em operação de pátio.
- Armazéns e silos ao longo da orla portuária.
- Trabalhadores portuários e equipamentos de sinalização (coletes, EPIs).

---

## 5. Vocabulário técnico útil para legendas (PT-BR / EN)

| Português | Inglês |
|---|---|
| Navio porta-contêineres | Container ship |
| Navio graneleiro | Bulk carrier |
| Navio-tanque | Tanker |
| Cais / berço | Berth / quay |
| Guindaste de cais (portêiner) | Ship-to-shore (STS) crane |
| Transtêiner | Rubber-tyred gantry (RTG) crane |
| Contêiner | Container |
| Pátio de contêineres | Container yard |
| Rebocador | Tugboat |
| Prático / praticagem | Harbor pilot / pilotage |
| Terminal de granéis | Bulk terminal |
| Armazém | Warehouse |
| Silo | Silo |
| Defensa | Fender |
| Cabeço de amarração | Bollard |
| Empilhadeira reach stacker | Reach stacker |
| Zona alfandegada | Customs area |

---

## 6. Fontes
- [CODEBA — Porto de Salvador](https://www.codeba.gov.br/eficiente/sites/portalcodeba/pt-br/porto_salvador.php)
- [CODEBA — Arrendamentos em operação](https://www.codeba.gov.br/eficiente/sites/portalcodeba/pt-br/site.php?secao=porto_salvador_arrendamentos_operacao)
- [Wilson Sons — Tecon Salvador](https://wilsonsons.com.br/pt-br/teconsalvador/)
- [Wilson Sons — Infraestrutura do Tecon Salvador](https://wilsonsons.com.br/pt-br/teconsalvador/infraestrutura/)
- [Agência Gov — Porto de Salvador se moderniza](https://agenciagov.ebc.com.br/noticias/202405/maiores-navios-porta-conteineres-do-mundo-com-carga-total-vao-atracar-no-porto-de-salvador)
- [Aratu On — Portos da Bahia](https://aratuon.com.br/infraestrutura/portos-da-bahia-como-os-terminais-impulsionam-o-estado/)

---

## 7. Próximos passos sugeridos
- [ ] Definir categorias/classes de objetos relevantes para anotação (ex: navio, contêiner,
      guindaste, rebocador, armazém, silo, cais).
- [ ] Levantar fontes de imagens do Porto de Salvador (fotos públicas, CODEBA, imprensa,
      satélite) respeitando direitos de uso.
- [ ] Definir formato e granularidade das legendas (descrição geral da cena vs. objetos
      específicos).
- [ ] Validar vocabulário técnico com especialista do domínio (se disponível).
