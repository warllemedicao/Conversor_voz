# Relatorio Voz_Noslen F5-TTS ONNX (Modo Lite)

Este documento registra a intencao atual do notebook Kaggle e as correcoes aplicadas em 2026-06-17 para suporte ao **Modo Lite (Cloud Run)**.

## Objetivo

Gerar, no Kaggle, um pacote ONNX otimizado para execução no Cloud Run (Modo Lite) da voz neural treinada `Voz_Noslen`.

O pacote deve:
- Seguir o contrato tecnico do motor SuperVoz-F5-Lite.
- Preservar o checkpoint original `.pt` para metadados.
- Exportar o ONNX com as entradas de controle (`speed`, `n_steps`).

## Arquivos alterados

- `kaggle/f5_tts_onnx_packager_kaggle.py`
- `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb`
- `kaggle/README_kaggle.md`
- `kaggle/RELATORIO_VOZ_NOSLEN_ONNX.md`

## Formato final do pacote (Modo Lite)

Estrutura obrigatoria:

```text
/onnx_package_name/
├── onnx/
│   └── f5_tts_transformer_core.onnx
├── model/
│   ├── model_2000.pt
│   └── vocab.txt
└── reference/
    └── referencia_voz.wav
```

## Contrato tecnico do ONNX Lite

* **Arquivo**: `f5_tts_transformer_core.onnx`
* **Opset**: 17
* **Entradas (Inputs)**:
    - `text_ids`: IDs do texto alvo.
    - `text_lengths`: Comprimento do texto alvo.
    - `ref_text_ids`: IDs do texto de referencia.
    - `ref_text_lengths`: Comprimento do texto de referencia.
    - `speed`: Fator de velocidade.
    - `n_steps`: Numero de passos de inferencia (NFE).
* **Saída (Output)**:
    - `audio`: Waveform gerada (eixos dinâmicos).

## Correcoes aplicadas (2026-06-17)

### 1. Atualização para Modo Lite
Migração do antigo "Modo Turbo" para o "Modo Lite" compatível com Cloud Run. O ONNX agora recebe IDs de texto e parâmetros de controle diretamente.

### 2. Preservação de Metadados
O arquivo `model_2000.pt` é mantido no pacote final, pois o backend Lite o utiliza para ler a configuração da arquitetura `F5TTS_v1_Base` antes de carregar o grafo ONNX.

### 3. Simplificação do Pacote
Remoção de scripts e manifestos extras que não são utilizados pelo motor Lite, focando na estrutura de pastas `onnx/`, `model/` e `reference/`.

## Variaveis principais

```text
HF_SOURCE_URL=https://huggingface.co/buckets/warllem/Voz_Noslen
HF_VOICE_DIR=voices/v_minha_voz_f5_tts_ptbr
HF_UPLOAD_REPO_ID=warllem/Voz_Noslen_ONNX
HF_DOWNLOAD_MODE=essential
F5_ONNX_QUANTIZE=0
F5_ONNX_RUN_CPU_TEST=1
```

## Validacoes locais

Foram feitas validacoes sem baixar modelos grandes:

```bash
python -m py_compile kaggle/f5_tts_onnx_packager_kaggle.py
python kaggle/f5_tts_onnx_packager_kaggle.py --help
python -m json.tool kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb
```

O teste real de exportacao ONNX depende do Kaggle com Internet e dependencias instaladas.
