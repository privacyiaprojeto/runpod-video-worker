# runpod-video-worker

Worker Serverless RunPod para prova técnica do pipeline de vídeo / FaceSwap.

Fluxo:

```txt
source_image_url + target_video_url
→ download
→ InsightFace FaceAnalysis + INSwapper
→ processamento frame a frame
→ MP4 final
→ retorno base64 ou URL R2 opcional
```

## Segurança operacional

Use apenas assets licenciados/consentidos. O worker exige `safety_mode=licensed_or_consented_assets_only` ou `consent_confirmed=true` no payload.

## Arquivos

```txt
Dockerfile
handler.py
requirements.txt
test_input.json
.dockerignore
```

## Modelo necessário

Coloque o modelo de FaceSwap em:

```txt
/runpod-volume/models/inswapper_128.onnx
```

ou configure:

```env
SWAPPER_MODEL_URL=https://sua-url-privada/inswapper_128.onnx
```

Para uso comercial, valide a licença do modelo/ferramenta escolhido.

## Variáveis principais

```env
MODEL_DIR=/runpod-volume/models
SWAPPER_MODEL_PATH=/runpod-volume/models/inswapper_128.onnx
SWAPPER_MODEL_URL=
DEFAULT_MAX_SECONDS=12
MAX_DOWNLOAD_MB=300
MAX_BASE64_RETURN_MB=80
PRESERVE_AUDIO=true
SWAP_ALL_FACES=false
RETURN_BASE64_DEFAULT=true
```

## R2 opcional no worker

O backend já consegue receber base64 e salvar no R2. Para vídeos maiores, você pode fazer upload direto no worker:

```env
R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
R2_PUBLIC_BASE_URL=https://seu-dominio-r2.com
R2_PREFIX=faceswap/tmp
```

## Build local

```bash
docker build -t runpod-video-worker:0.1.0 .
```

## Build e push para GHCR

```bash
docker build -t ghcr.io/SEU_USUARIO/runpod-video-worker:0.1.0 .
docker push ghcr.io/SEU_USUARIO/runpod-video-worker:0.1.0
```

## Payload de teste

```json
{
  "input": {
    "source_image_url": "https://seu-r2/avatar-sofia.jpg",
    "target_video_url": "https://seu-r2/video-base-curto.mp4",
    "safety_mode": "licensed_or_consented_assets_only",
    "max_seconds": 8,
    "return_base64": true
  }
}
```

## Resposta esperada

Base64:

```json
{
  "video_base64": "...",
  "mime_type": "video/mp4",
  "extension": "mp4",
  "size_bytes": 123456,
  "elapsed_ms": 45000
}
```

ou URL R2:

```json
{
  "video_url": "https://.../faceswap/tmp/file.mp4",
  "url": "https://.../faceswap/tmp/file.mp4",
  "mime_type": "video/mp4",
  "extension": "mp4",
  "size_bytes": 123456,
  "elapsed_ms": 45000
}
```
