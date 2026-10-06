# Comparativo das variantes

Referencias: labels.jsonl (split test) — rotulador `claude-opus-5-5`. Modelo gold: `gemini-3.8-flash` (gemini). SLM: `Qwen/Qwen3-VL-2B-Instruct`.

| variante | o que e |
|---|---|
| base | Qwen3-VL-2B original, zero-shot (sem treino) |
| pretrain | pre-treino continuado de dominio (so o conector visao->LLM) |
| finetune | fine-tuning completo (conector + LLM) |
| lora | LoRA no LLM + conector |
| qlora | QLoRA (LLM 4 bits) + conector |
| gold | VLM grande via API (referencia externa) |

## Metricas

| variante | N | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | BERTScore-F1 | CLIPScore | RefCLIPScore | Distinct-1 | Distinct-2 | Tam.medio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 15 | 0.3844 | 0.0268 | 0.2317 | 0.1172 | 0.6532 | 0.8114 | 0.8263 | 0.3325 | 0.6649 | 25.6667 |
| pretrain | 15 | 0.5096 | 0.0859 | 0.3038 | 0.2490 | 0.7032 | 0.7230 | 0.7979 | 0.3567 | 0.6722 | 20.9333 |
| finetune | 15 | 0.5277 | 0.0667 | 0.3067 | 0.2899 | 0.7111 | 0.7006 | 0.7748 | 0.3301 | 0.7010 | 20.4000 |
| lora | 15 | 0.5911 | 0.1419 | 0.3587 | 0.5069 | 0.7369 | 0.6649 | 0.7706 | 0.3227 | 0.6745 | 20.8667 |
| qlora | 15 | 0.5981 | 0.1544 | 0.3822 | 0.5324 | 0.7487 | 0.6779 | 0.7813 | 0.3258 | 0.6712 | 20.6667 |
| gold | 15 | 0.6606 | 0.2253 | 0.4404 | 0.8700 | 0.7897 | 0.7063 | 0.8063 | 0.3303 | 0.6731 | 21.8000 |

> CIDEr calcula o IDF no proprio conjunto avaliado: com N < ~100 o valor e instavel.

## Custo de treino

| variante | ponto de partida | quantizacao | params treinaveis | % | VRAM pico (GB) | tempo (min) | melhor val loss | disco (MB) |
|---|---|---|---|---|---|---|---|---|

## Exemplos

### psa_0002

- **referencia (Claude):** Spreader de um guindaste portuário encaixa-se sobre o teto de um contêiner empilhado durante a movimentação de contêineres.
- **base:** Um gancho de carregamento vermelho, com número 75, opera sobre um contêiner marítimo de cor vermelha, transportando uma carga de mercadorias em um terminal portuário.
- **pretrain:** Contêiner de 40 pés é levantado por um guindaste de contêiner, posicionando-se sobre outro contínuo.
- **finetune:** Ponteiro de um guindaste eleva um contêiner sobre outro, posicionando-o sobre uma fileira de contêiners no pátio de um terminal.
- **lora:** Contêiner suspenso pelo spreader de um guindaste sobre outro contêiner, com pilhas de contêiners ao redor no pátio de terminal.
- **qlora:** Pegada de um spreader de um guindaste sobre um contêiner, com o spreader apoiado no topo de outro contínuen, ao lado de outros contêiners no pátio.
- **gold:** Spreader posiciona-se sobre um contêiner durante operação de movimentação de carga a bordo de um navio porta-contêineres.

### psa_0010

- **referencia (Claude):** Reach stacker sem carga está estacionado no cais, ao lado de feixes de barras de aço armazenados e de caminhões parados junto à água.
- **base:** O porto está em atividade, com um guindaste de carga vermelho em operação, transportando barras de aço, enquanto outros veículos e equipamentos estão prontos para a movimentação de merc
- **pretrain:** Ponteiro portuário está estacionado no pátio de um porto, ao lado de outros veículos de apoio, com pilhas de barras metálicas empilhadas no chão.
- **finetune:** Ponteiro portuário está estacionado no pátio, com sua lança estendida, ao lado de outros veículos de apoio e pilhas de barras metálicas.
- **lora:** Pátio de terminal de contêineres com reach stacker, pilhas de barras metálicas, carretas e veículos de apoio junto à água.
- **qlora:** Pátio de terminal portuário com reach stacker, carretas de guindaste e pilhas de barras metálicas no pátio, junto à água.
- **gold:** Uma reach stacker com spreader estendido permanece estacionada no cais, próxima a caminhões e feixes de carga geral dispostos no solo.

### psa_0011

- **referencia (Claude):** Reach stacker ergue um contêiner-tanque com o spreader enquanto um trabalhador de capacete, a pé, sinaliza a manobra com o braço.
- **base:** Um guindaste de carga Taylor 9975C transporta um tanque de gás comprimido sobre uma plataforma de armazenamento de tanques de petróleo da empresa Seaco, em uma operação de
- **pretrain:** Um carregador portuário Taylor 9975C eleva um conjunto de tanques de petróleo em um pátio de armazenamento.
- **finetune:** Porte de um tanque térmico por meio de um portênter com o braço estendido sobre um pátio de armazenamento de tanques.
- **lora:** Pátio de armazenagem de tanques de combustível com reach stacker em operação, um trabalhador caminhando no pátio e outros tanques empilhados.
- **qlora:** Pátio de armazenagem de contêineres-tanque com um reach stacker transportando um dos tanques, um trabalhador com capacete e um pilar de proteção no chão.
- **gold:** Reach stacker iça um contêiner-tanque para empilhamento no pátio, orientada por um trabalhador portuário com capacete junto a outros contêineres-tanque armazenados.

### psa_0017

- **referencia (Claude):** Navio porta-contêineres carregado e equipado com dois guindastes de bordo está parado na água, cercado por pequenas embarcações de madeira.
- **base:** O porto está ativo com um grande navio de contêineres da marca Delmas, sendo movido por uma grande grua de carga, enquanto outros barcos navegam na água ao redor.
- **pretrain:** Contêineres empilhados no convés de um portacontêiner, com pílotos e escorregadores em posição, enquanto o navio se aproxima do cais.
- **finetune:** Contêineres estão empilhados no deck de um navio porta-contedores, que está atracado ao lado de um cais, com um guindaste de pórtico sobre o convés.
- **lora:** Navio porta-contêineres atracado junto a um cais, com lanças de guindaste erguidas sobre o convés e navios menores atracados no cais ao fundo.
- **qlora:** Navio porta-contêineres atracado no cais, com guindaste de bordo erguido sobre o convés e um navio menor ancorado ao lado.
- **gold:** Navio porta-contêineres com guindastes de bordo e contêineres empilhados sobre o convés encontra-se atracado ao cais por cabos de amarração.

### psa_0028

- **referencia (Claude):** Navio porta-contêineres está atracado e amarrado ao cais por cabos, sob as lanças baixadas dos portêineres do terminal.
- **base:** Em um porto sob céu nublado, um navio de carga azul está atracado em um cais, com gruas de grande porte e equipamentos de manuseio de contêineres,
- **pretrain:** Um navio porta-contêineres está atracado em um cais, com um guindaste portuário sobre o convés e vários contêiners empilhados ao longo do cais.
- **finetune:** Contêineres são empilhados no cais, enquanto um navio está atracado ao pôr do sol.
- **lora:** Navio porta-contêineres está atracado junto ao cais, com o spreader de um portêiner posicionado sobre o convés.
- **qlora:** Terminal de contêineres com navio atracado ao lado do cais, portêinê e equipamentos de armazenagem ao fundo.
- **gold:** Um navio porta-contêineres carregado permanece atracado ao cais sob portêineres em um terminal de contêineres, com carretas e contêineres posicionados na área operacional.

### psa_0035

- **referencia (Claude):** Caminhão transporta um contêiner de 40 pés sobre carreta porta-contêiner ao longo do cais, sob a estrutura de um portêiner.
- **base:** Um caminhão de carga Maersk transporta um contêiner de mercadorias em um porto, com uma grua de carga grande e uma embarcação ao fundo, sob um céu claro.
- **pretrain:** Um caminhão de contêiner está estacionado no pátio de um porto, com um contenedor de 20 pés sobre o carro, e um guindaste de 40 metros ao fundo
- **finetune:** Contêiner Maersk transportado por caminhão sobre o pátio de um terminal portuário, ao lado de um guindaste de contêineres e uma antena de comunicação.
- **lora:** Contêiner transportado por um caminhão está posicionado no pátio junto ao guindaste de pórtico de um navio porta-contêineres.
- **qlora:** Pátio de terminal de contêineres com um caminhão carregando um contínuo, um guindaste de pórtico sobre o navio e o radar de um navio atracado ao fund
- **gold:** Caminhão com carreta porta-contêiner transporta um contêiner de 40 pés pelo cais, passando sob a estrutura de um portêiner em um terminal de contêineres.

### psa_0053

- **referencia (Claude):** Trator de pátio com operador na cabine reboca uma carreta porta-contêiner vazia pelo pátio de um terminal de contêineres.
- **base:** Um caminhão de transporte de contêineres, modelo 23, move-se sobre uma pista de carga em um porto, carregando contêiners de diferentes operadores, incluindo EMT e Tex
- **pretrain:** Um caminhão de contêineres de 23 polegadas avança por uma pista de estacionamento, com um caminho de trilhos e um conjunto de quatro rodas.
- **finetune:** Contêineres empilhados ao longo da pista, um caminhão-piloto de pórtico com trator e caçamba de contêiner transportando um trailer de carga.
- **lora:** Contêineres estão empilhados em três níveis ao longo do pátio pavimentado, com um reach stacker de 18 pés posicionado no centro da imagem.
- **qlora:** Contêiner transportado por um caminhão porta-contêiner atracado ao lado de pilhas de contêineres no pátio de um terminal.
- **gold:** Caminhão tracionando carreta porta-contêiner vazia realiza manobra no pátio de contêineres, diante de contêineres empilhados.

### psa_0054

- **referencia (Claude):** Transtêineres com spreaders suspensos estão posicionados sobre as vias de um pátio de contêineres, entre blocos de contêineres empilhados.
- **base:** Cargas de contêineres verdes e azuis estão sendo movimentadas por guindastes de grande porte em um porto industrial sob um céu nublado.
- **pretrain:** Ponto de estacionamento de contêineres em um terminal portuário, com dois guindastes de pílulas e pilares de apoio ao longo do pátio.
- **finetune:** Pátio de um terminal de contêineres, com dois guindastes portêineros posicionados entre pilhas de caixas empilhadas.
- **lora:** Pátio de terminal de contêineres com dois spanniers, pátio pavimentado com linhas de trânsito e pilhas organizadas ao longo das laterais.
- **qlora:** Pátio de terminal de contêineres com pátio pavimentado, pilhas de caixas empilhadas e dois portêiners com lanças suspensas.
- **gold:** Transtêineres operam sobre pilhas de contêineres no pátio de contêineres de um terminal portuário, com spreaders suspensos prontos para movimentação de carga.
