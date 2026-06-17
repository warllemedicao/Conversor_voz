# Relatorio Voz_Noslen F5-TTS ONNX Turbo

Este documento registra a intencao atual do notebook Kaggle e as correcoes aplicadas em 2026-06-17.

## Objetivo

Gerar, no Kaggle, um pacote ONNX Turbo da voz neural treinada `Voz_Noslen`.

O pacote deve:

- baixar somente os arquivos essenciais da voz F5-TTS;
- preservar a qualidade usando precisao original por padrao;
- exportar um ONNX Turbo com DiT + Euler + Vocos;
- remover do pacote final a arvore antiga de treino `f5_tts_original/`;
- validar o pacote com teste CPU;
- enviar o resultado para um Model Repo normal do Hugging Face.

## Arquivos alterados

- `kaggle/f5_tts_onnx_packager_kaggle.py`
- `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb`
- `kaggle/conversor_voz_requirements_kaggle.txt`
- `kaggle/README_kaggle.md`
- `kaggle/RELATORIO_VOZ_NOSLEN_ONNX.md`

## Formato final do pacote

O pacote final nao inclui mais `f5_tts_original/`.

Estrutura esperada:

```text
model/
  model_2000.pt
  vocab.txt
reference/
  referencia_voz.wav
  reference_text.txt
onnx/
  f5_tts_turbo_original_precision.onnx
scripts/
  test_package_cpu.py
manifest.json
package_metadata.json
onnx_export_report.json
```

Se `F5_ONNX_QUANTIZE=1`, o pacote tambem pode conter:

```text
onnx/f5_tts_turbo_int8.onnx
```

## Politica de qualidade

A versao anterior apresentava INT8 como caminho principal. Isso foi corrigido.

Agora:

- o ONNX principal e `f5_tts_turbo_original_precision.onnx`;
- INT8 e opcional e desativado por padrao;
- o manifesto registra que a prioridade e preservar a voz neural treinada;
- o checkpoint original, vocabulario e referencia continuam no pacote.

Essa decisao evita perda perceptivel de qualidade causada por quantizacao agressiva.

## Contrato tecnico do ONNX Turbo

O ONNX Turbo recebe tensores ja preparados:

```text
x
cond
text
time_steps
mask
```

E retorna:

```text
audio
```

Ele encapsula:

- Transformer/DiT;
- loop Euler;
- Vocos.

Ele nao substitui todo o frontend de inferencia do F5-TTS. O backend ainda precisa preparar texto, referencia, condicionamento, noise, mascara e passos de tempo.

## Correcoes aplicadas

### 1. Notebook com instalacao quebravel

Problema:

```text
!pip install -q f5-tts>=1.1.9 ...
```

O shell pode interpretar `>` como redirecionamento.

Correcao:

- o notebook agora escreve `conversor_voz_requirements_kaggle.txt` em `/kaggle/working`;
- instala com `python -m pip install -q -r ...`;
- usa a lista completa de dependencias, incluindo `onnxscript`.

### 2. Script embutido divergente

Problema:

O notebook tinha uma copia grande do script. Se o `.py` fosse alterado e o notebook nao fosse regenerado, o Kaggle rodaria codigo antigo.

Correcao:

- o notebook foi regenerado a partir de `f5_tts_onnx_packager_kaggle.py`;
- a copia embutida agora corresponde ao script atual.

### 3. Uso de `torch` antes do import

Problema:

A classe do wrapper usava `torch.nn.Module` no escopo global, antes de `torch` estar importado.

Correcao:

- o wrapper foi movido para dentro de `export_f5_core_to_onnx`, apos `import torch`.

### 4. Pacote final com formato antigo de treino

Problema:

O empacotador movia o snapshot baixado para `f5_tts_original/` dentro do pacote final.

Correcao:

- o snapshot baixado fica apenas como area de trabalho;
- o pacote final recebe somente runtime minimo;
- os metadados registram `legacy_training_tree_included: false`.

### 5. Smoke test procurando ONNX no lugar errado

Problema:

`test_package_cpu.py` procurava `model/*.onnx`, mas os arquivos eram gerados em `onnx/`.

Correcao:

- o teste agora procura `onnx/*.onnx`.

### 6. Logger ausente no script de teste

Problema:

`test_package_cpu.py` chamava `LOGGER.warning` e `LOGGER.info`, mas `LOGGER` nao existia.

Correcao:

- o script gerado agora configura `logging` e define `LOGGER`.

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
