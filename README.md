# Privacy IA — RunPod Wan 2.1 / ComfyUI Video Worker

Worker Serverless de vídeo para o contrato canônico `privacy-production-spec-v1`.

## Escopo do Pivot Stage 2

O runtime anterior foi substituído por uma arquitetura ComfyUI headless, com workflows versionados para:

- `wan-2.1-i2v` — imagem aprovada do Cofre KYC para vídeo;
- `wan-2.1-v2v` — vídeo base privado + imagem aprovada do Cofre KYC;
- saída privada, obrigatoriamente marcada como `qa_required`;
- nenhum upload público persistente;
- telemetria estruturada em JSON;
- testes locais sem GPU.

## Fluxo

```text
privacy-production-spec-v1
  → validação fail-closed do contrato e das flags de segurança
  → seleção da referência aprovada em identity.actors[].references
  → download temporário por URL assinada
  → injeção nos nós do workflow ComfyUI
  → execução Wan 2.1
  → MP4 privado em base64 ou R2 privado
  → backend registra Master em qa_pending
```

## Workflows versionados

```text
workflows/wan-2.1-i2v-v1.json
workflows/wan-2.1-v2v-v1.json
```

Cada arquivo é um envelope `privacy-comfyui-workflow-v1` contendo:

- `prompt`: grafo no API format do ComfyUI;
- `bindings`: mapa entre o contrato do backend e os inputs dos nós;
- `output_nodes`: nós que publicam o MP4;
- `engine` e `workflow_version`.

O backend pode fornecer `comfyui.graph` para substituir o grafo em uma futura versão, mantendo os bindings empacotados ou enviando bindings próprios no envelope.

## Modelos no RunPod Network Volume

Os pesos não são incluídos na imagem Docker. Monte um Network Volume em `/runpod-volume` e disponibilize:

```text
/runpod-volume/models/
├── diffusion_models/
│   ├── wan2.1_i2v_480p_14B_fp16.safetensors
│   └── wan2.1_vace_14B_fp16.safetensors
├── text_encoders/
│   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
├── vae/
│   └── wan_2.1_vae.safetensors
└── clip_vision/
    └── clip_vision_h.safetensors
```

Os nomes podem ser alterados por variáveis de ambiente. O worker falha com mensagem clara caso algum modelo obrigatório esteja ausente.

## Segurança de entrada

- aceita somente `http`/`https`;
- bloqueia IPs privados, loopback, link-local e reservados;
- revalida cada redirecionamento;
- permite allowlist por `MEDIA_ALLOWED_HOSTS`;
- limita tamanho de imagens, vídeos e saída;
- a imagem de identidade precisa existir em `identity.actors[].references`;
- ignora `conditioning.source_image_url` quando não corresponde a uma referência aprovada;
- exige todas as flags de segurança do contrato como `true`.

Em produção, configure `MEDIA_ALLOWED_HOSTS` com os hosts oficiais usados pelas Signed URLs do R2.

## Saída privada

Modo recomendado:

```env
OUTPUT_MODE=private_r2
```

O objeto é enviado com cache privado e o worker retorna uma Signed URL curta, `r2_bucket` e `r2_key`. Não existe suporte a URL pública persistente.

## RunPod Cached Models (opt-in)

O modo padrão permanece `MODEL_SOURCE_MODE=network_volume`. O modo
`cached_model` não baixa pesos: ele exige `CACHED_MODEL_ID=org/repo` e uma
`CACHED_MODEL_REVISION` exata já montada em
`/runpod-volume/huggingface-cache/hub/models--org--repo/snapshots/REVISION`.
O snapshot precisa conter os diretórios `diffusion_models`, `text_encoders`,
`vae` e `clip_vision`, com os mesmos arquivos obrigatórios descritos acima. No
top-level, somente os metadados opcionais `.gitattributes` e `README.md` também
são aceitos.

Nesse modo, `MODEL_ROOT` usa por padrão `/tmp/privacy-models`: as quatro
categorias read-only são symlinks para o snapshot e `loras` é um diretório
efêmero gravável. `RUNTIME_ROOT` usa `/tmp/privacy-wan-runtime`. A configuração
falha no startup se o backend global de one-shot lock não for explicitamente
`IDENTITY_ONE_SHOT_LOCK_BACKEND=r2` ou se o R2 privado não estiver configurado.

`EPHEMERAL_MIN_FREE_GB=20` é o piso conservador padrão para adapter, mídias e
outputs temporários. Ajustes precisam ser explícitos e o worker sempre verifica
o espaço antes de downloads pesados. Nenhum modelo ou token HF é incluído na
imagem.

## Build

```bash
docker build -t runpod-video-worker:2.0.0 .
```

A imagem instala:

- PyTorch/CUDA;
- ComfyUI headless;
- ComfyUI-VideoHelperSuite;
- FFmpeg e FFprobe;
- SDK RunPod;
- adaptador canônico do Privacy IA.

Os argumentos `COMFYUI_REF` e `VIDEO_HELPER_REF` permitem fixar revisões homologadas:

```bash
docker build \
  --build-arg COMFYUI_REF=v0.27.0 \
  --build-arg VIDEO_HELPER_REF=main \
  -t runpod-video-worker:2.0.0 .
```

Após o primeiro smoke de GPU aprovado, substitua `VIDEO_HELPER_REF=main` pelo commit exato homologado.

## Testes sem GPU

```bash
python -m pip install -r requirements-dev.txt
python scripts/validate_workflows.py
python -m pytest
```

Os testes comprovam:

- parsing I2V e V2V do contrato canônico;
- exigência das referências aprovadas;
- normalização de dimensões e frames Wan;
- injeção de prompt, KYC, vídeo base, FPS e sampling nos workflows;
- execução mockada do `handler.py` sem ComfyUI/GPU;
- ausência das dependências e contratos do runtime anterior.

## Payloads de teste

```text
test_input.json       # I2V
test_input_v2v.json   # V2V
```

As URLs são exemplos e não são baixadas nos testes unitários.

## Limites conhecidos antes do smoke real

- os testes locais não carregam pesos nem executam CUDA;
- os nomes e inputs de custom nodes devem ser confirmados no build/smoke da imagem final;
- o workflow V2V v1 processa um ator principal por job e falha de forma controlada quando o contrato não oferece referência aprovada;
- produção real permanece bloqueada no backend até o endpoint RunPod ser atualizado e as flags serem habilitadas de forma controlada.
