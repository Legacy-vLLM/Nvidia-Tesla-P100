Utilizar o  vLLM na NVIDIA Tesla P100 com LoRA, Paged Attention e GPTQ ativados.

Neste repositório você encontrara o vLLM Legacy compatível com a Nvidia Tesla P100 16GB HBM2. 

A Nvidia Tesla P100 de 16GB possui um hardware impressionante sua memoria funciona a ~730 GB/s de largura de banda de memória (HBM2) a 4096 bits (sim 4096 e não 384 bits.)

No entanto, ela foi lançada em 2016 um ano antes do lançamento de placas com “Tensor Cores“ (tecnologia atual) com o surgimento de Tensor Cores a Nvidia tesla p100 acabou ficando abandonada em termos de software, foi por isso que surgiu este projeto.

O Legacy vLLM P100 resolve este gargalo. Portamos as otimizações cruciais do vLLM Oficial de volta para a arquitetura Pascal nesta versão customizada, permitindo que você reative suas P100 e sirva modelos modernos com performance corporativa e compatibilidade total.

MODELOS TESTADOS:

MODELO
THROUGHPUT DECODE - 1 GPU
Qwen/Qwen3-1.7B-GPTQ-Int8
107 tk/s
Qwen/Qwen 2.5 1.5B
71 tk/s
Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8
45 tk/s
microsoft/Phi-3.5-vision-instruct
40 tk/s
JunHowie/Qwen3-8B-GPTQ-Int8
45 tk/s






O Status do Legacy-vLLM na P100 (Compute 6.0)
Recurso
Funcionando na P100?
Detalhes Técnico
xFormers Attention
ativado
Kernels de atenção do xFormers rodando estavelmente com vLLM moderno. 
Paged Attention
ativado

Quando passa de 5 para 20 ou 50 requisições, o valor total de tokens/s ( througput generation) sobe drasticamente.


CUDA Graphs
ativado
Totalmente suportado em nível de driver CUDA 12.4 na P100. Ajuda imensamente a mitigar o overhead em lotes (batch sizes).
API OpenAi 
ativado
Funcionando com métricas de desempenho formatadas para o Prometheus.
KV Cache na GPU
ativado
O gerenciamento de memória HBM2 (que na P100 tem excelentes ~730 GB/s de banda) funciona perfeitamente com a alocação dinâmica do vLLM.
Batching
ativado
Com a alocação dinâmica do vLLM.
Embedding
ativado
Funcionando.
LoRA
ativado
Funcionando.


Com o Paged Attention (via xFormers), Scheduler, CUDA Graphs e KV Cache ativados na P100, atingimos o topo absoluto do que o hardware da P100 pode entregar. É o ápice da portabilidade e um excelente produto para reviver esses hardwares em datacenters pelo mundo.


Recursos de Classe Corporativa Integrados
vLLM Engine V0 Port: Conversão automática e transparente de modelos nativos GPTQ para float16 (half)... para máxima eficiência matemática nos núcleos Pascal.
XFormers Backend Ativo: Substitui o FlashAttention-2, garantindo baixa latência de atenção.
CUDA Graphs: Captura de grafos estáticos de execução ativa, reduzindo o overhead da CPU durante o decoding no warmup.
Modelos GPTQ: Conversão automática e transparente de modelos nativos GPTQ para float16 (half) Bfloat16 para float16 para máxima eficiência matemática nos núcleos Pascal.
API OpenAI Compatível: Pronto para integrar diretamente com LangChain, LlamaIndex, LiteLLM ou qualquer aplicação existente que aponte para a porta :8000.
 Benchmarks Reais (Tesla P100 PCIe 16GB)
Cenário de Teste: Qwen/Qwen2.5-1.5B-Instruct
Throughput (requisição única) Geração: 71.9 tokens/segundo (Medido diretamente na engine, livre de latências de rede HTTP e serializações JSON).
Quando passa de 5 para 20 ou 50 requisições, é aqui que entra o poder do Paged Attention o valor total de tokens/s vai subir drasticamente. O vLLM consegue processar múltiplos pedidos ao mesmo tempo na GPU (via continuous batching) na suas 4096 bits.
Se você possui uma Tesla P100 teste hoje mesmo você vai se surpreender com o desempenho, fornecemos a imagem Docker link abaixo.

Isso é suficiente para servir:
Chatbots locais; 
RAG; 
agentes; 
automações; 
APIs privadas; 
integrações OpenWebUI ou LiteLLM…


Como Rodar o Legacy-vLLM Docker Teste (Trial de 12 Horas)
Nós disponibilizamos uma versão de avaliação gratuita para você validar a estabilidade do sistema diretamente no seu hardware .
1. Prepare seu ambiente Docker
Certifique-se de possuir o Docker instalado (Se já possui ir para o Passo 2)
Linux
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
Windows
https://www.docker.com/get-started/
2. Instale o Nvidia Container Toolkit inicialização
Certifique-se de possuir o nvidia-container-toolkit devidamente configurados no seu host,
o Docker moderno utiliza uma nova especificação chamada CDI (Container Device Interface) para se comunicar com as GPUs.
Linux:
Passo 1: Atualizar os repositórios do NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \ && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \ sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \ sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

Passo 2: Instalar o Toolkit
Atualize o apt e instale o pacote nvidia-container-toolkit:
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
Passo 3:
Abra o arquivo no terminal com o editor nano (ou seu editor de preferência):
sudo nano /etc/docker/daemon.json
Se o arquivo estiver vazio, cole o conteúdo abaixo. Se ele já tiver alguma coisa, adicione a chave "runtimes" e defina o "default-runtime" para que a NVIDIA gerencie os contêineres:
JSON
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
Salve o arquivo (no Nano, aperte Ctrl + O, depois Enter para confirmar, e Ctrl + X para sair).
Passo 4: Criar a pasta do CDI e gerar os dispositivos
Como o Docker moderno usa o CDI para mapear a sua GPU P100, precisamos garantir que o diretório padrão exista e que o arquivo YAML com os metadados da sua placa de vídeo seja criado manualmente:
Crie a pasta padrão de CDI no sistema:
sudo mkdir -p /etc/cdi
Force o utilitário do toolkit a mapear as GPUs instaladas na sua máquina física e escrever o arquivo de configuração:
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
Passo 5: Reiniciar o serviço do Docker
Agora que as configurações manuais estão no lugar e o arquivo do CDI foi gerado, reinicie o daemon do Docker para carregar o novo runtime:
sudo systemctl restart docker

5. Solicite sua chave de Teste de 12 Horas
Obtenha sua chave de teste no formulário:
https://docs.google.com/forms/d/e/1FAIpQLSeJw9ZslKvIkDJtKjlVbirIJdfXlDiFhXMoQPivBWII_Ok8wQ/viewform?usp=header

6. Baixe a imagem docker Legacy-Vllm

docker pull legacyvllm/legacy-vllm-tesla-p100-openai-lora:310726


Execute o comando de inicialização no terminal:
docker run --gpus all \
  -e LEGACY_VLLM_LICENSE_KEY=" **Solicite sua chave trial e coloque aqui**" \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
legacy-vllm\p100-openai-lora:300726 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048

Segurança e Modelo de Licenciamento
Para atender às rígidas políticas de segurança corporativa (incluindo datacenters fechados e ambientes air-gapped sem acesso à internet), o nosso sistema de licenças utiliza criptografia assimétrica offline:
A imagem Docker contém apenas a nossa chave pública.
A verificação do tempo de expiração é calculada localmente na inicialização do contêiner.
Nenhum dado é enviado para servidores externos. Sua privacidade e conformidade com a LGPD/GDPR são totalmente preservadas.

Licenciamento Comercial
Gostou do teste? Oferecemos licenças mensais ou anuais (nós geramos chaves para o seu cluster de GPUs).

Para adquirir sua chave ou propostas comerciais, fale conosco: https://docs.google.com/forms/d/e/1FAIpQLSeJw9ZslKvIkDJtKjlVbirIJdfXlDiFhXMoQPivBWII_Ok8wQ/viewform?usp=header







