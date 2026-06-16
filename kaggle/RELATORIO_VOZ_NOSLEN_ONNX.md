# Relatorio do empacotador Voz_Noslen F5-TTS ONNX

Este documento registra o que foi implementado no empacotador Kaggle, os erros encontrados durante a execucao no Kaggle e as correcoes aplicadas.

## Objetivo

Criar um notebook Kaggle autocontido para:

- baixar/copiar a voz F5-TTS `Voz_Noslen` do Hugging Face Buckets;
- preservar os arquivos originais do treinamento;
- gerar um pacote novo em `/kaggle/working/voz_noslen_onnx_package`;
- exportar um ONNX experimental do nucleo Transformer/DiT do F5-TTS;
- enviar o pacote final para um Model Repo normal do Hugging Face em uma pasta nova.

## Arquivos criados ou alterados

- `kaggle/f5_tts_onnx_packager_kaggle.py`: script principal de download, empacotamento, exportacao ONNX e upload.
- `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb`: notebook Kaggle autocontido; ele recria o script e requirements em `/kaggle/working`.
- `kaggle/conversor_voz_requirements_kaggle.txt`: dependencias do Kaggle, incluindo `onnx`, `onnxscript` e `onnxruntime`.
- `kaggle/README_kaggle.md`: instrucoes resumidas de uso.

## Estrutura usada

Origem:

```text
https://huggingface.co/buckets/warllem/Voz_Noslen
```

Voz escolhida:

```text
voices/v_minha_voz_f5_tts_ptbr
```

Destino padrao:

```text
warllem/Voz_Noslen_ONNX
```

Pasta criada no destino:

```text
onnx_packages/voz_noslen_f5tts_onnx_<data_hora>
```

## Erro 1: repo_id nao encontrado

Erro:

```text
RepositoryNotFoundError: 404 Client Error
https://huggingface.co/api/models/warllem/Voz_Noslen/revision/main
```

Causa:

O link informado e um Hugging Face Buckets (`/buckets/...`), nao um Model Repo normal da API `huggingface_hub`. O primeiro script tentou usar `snapshot_download(repo_id="warllem/Voz_Noslen")`, mas esse repo nao existe como Model Repo.

Correcao aplicada:

- O notebook passou a usar `--source https://huggingface.co/buckets/warllem/Voz_Noslen`.
- O script ganhou suporte a origem `/buckets/` por leitura de links HTML.
- O upload passou a usar `--upload-repo-id warllem/Voz_Noslen_ONNX`, porque `HfApi.upload_folder` envia para Model Repo normal, nao para bucket.

## Erro 2: falta de espaco em disco

Erro:

```text
OSError: [Errno 28] No space left on device
```

Causa:

O script baixava muitos arquivos grandes e duplicados:

- `model_2000.pt`;
- `latest_checkpoint.pt`;
- `model_last.pt`;
- `base_checkpoint.safetensors`;
- arquivos `.tmp`;
- mais de uma pasta de voz.

Tambem havia duplicacao local, porque o snapshot era baixado em uma pasta e depois copiado para outra.

Correcao aplicada:

- Criado `--download-mode essential`.
- O modo `essential` ignora `.tmp`, checkpoints duplicados e baixa apenas:
  - manifesto;
  - checkpoint principal;
  - `vocab.txt`;
  - referencia da voz;
  - docs/configs pequenas;
  - arquivos pequenos da biblioteca base.
- A pasta baixada agora e movida para o pacote final, evitando duplicar o uso de disco.

## Erro 3: dependencia ONNX ausente

Erro:

```text
ModuleNotFoundError: No module named 'onnxscript'
```

Causa:

A versao do PyTorch instalada no Kaggle usa componentes novos do exportador ONNX que dependem de `onnxscript`.

Correcao aplicada:

- Adicionado `onnxscript>=0.1.0` ao requirements.
- O script agora mostra uma mensagem clara caso `onnxscript` esteja ausente.

## Erro 4: referencia de audio errada

Sintoma:

O script escolheu:

```text
libraries/f5_tts_ptbr_tharyck/audio_ref/F034-0960.wav
```

em vez de:

```text
voices/v_minha_voz_f5_tts_ptbr/data_reference/referencia_voz.wav
```

Causa:

A busca generica por `*.wav` encontrava primeiro um audio da biblioteca base.

Correcao aplicada:

- `find_reference_audio` agora prioriza `voices/v_minha_voz_f5_tts_ptbr/data_reference/referencia_voz.wav`.
- Somente se esse arquivo nao existir ele usa fallbacks genericos.

## Erro 5: torch.export falha com F5-TTS

Erro:

```text
GuardOnDataDependentSymNode
Could not extract specialized integer from data-dependent expression
Caused by: f5_tts/model/backbones/dit.py:91
max_seq_len = int(seq_len.max().item())
```

Causa:

O exportador novo do PyTorch (`torch.export`, usado pelo caminho ONNX moderno) nao consegue converter essa parte do F5-TTS, porque `seq_len.max().item()` depende de dado calculado durante a execucao.

Primeira tentativa de correcao:

- Trocar `torch.onnx.export` para `dynamo=False`, opset 18 e formas estaticas.

Resultado:

O erro continuou no Kaggle, indicando que o caminho novo ainda estava sendo usado ou que a versao do PyTorch nao respeitou completamente o caminho legado nessa chamada.

Correcao atual:

- Adicionado `PACKAGER_VERSION = "2026.06.15.5"` para confirmar no log se o Kaggle esta rodando o script atualizado.
- Criada funcao `legacy_onnx_export`.
- A exportacao agora tenta, nesta ordem:
  1. `torch.onnx.utils.export`, caminho legado explicito;
  2. `torch.onnx.export(..., dynamo=False)`;
  3. `torch.jit.trace(..., strict=False)` seguido de `torch.onnx.export`.
- O relatorio `onnx_export_report.json` registra `export_method` para mostrar qual caminho funcionou.

## Erro 6: dtype Float contra Half no exportador

Erro:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
```

Contexto observado no Kaggle:

```text
vocab :  /kaggle/working/voz_noslen_onnx_package/f5_tts_original/voices/v_minha_voz_f5_tts_ptbr/model/vocab.txt
token :  custom
model :  /kaggle/working/voz_noslen_onnx_package/f5_tts_original/voices/v_minha_voz_f5_tts_ptbr/model/model_2000.pt
```

Causa:

O checkpoint foi carregado com pesos em `float16` (`Half`), mas os tensores de exemplo usados para exportar o ONNX (`x`, `cond` e possivelmente `time`) eram criados em `float32` (`Float`). Durante uma camada `Linear`, o PyTorch recebeu entrada `Float` e peso `Half`, o que interrompeu a exportacao antes de gerar o ONNX.

Correcao aplicada:

- Atualizado `PACKAGER_VERSION` para `2026.06.15.6`.
- Criada funcao `infer_module_float_dtype` para detectar o dtype real dos pesos/buffers do modelo carregado.
- O wrapper ONNX agora converte entradas flutuantes (`x`, `cond`, `time`) para o dtype do modelo antes de chamar o Transformer.
- `text` continua inteiro e `mask` continua booleano, preservando os tipos esperados pelo F5-TTS.
- O relatorio `onnx_export_report.json` agora registra `model_compute_dtype`.

## Sobre qualidade de audio

Para preservar qualidade:

- nao quantizar o ONNX;
- manter FP32/FP16 original do checkpoint;
- usar o mesmo checkpoint treinado;
- usar o mesmo `vocab.txt`;
- manter arquitetura `F5TTS_v1_Base`;
- manter vocoder `vocos`;
- manter sample rate de 24000 Hz;
- manter a referencia de audio da propria voz treinada.

## Limitacao importante

O ONNX gerado e experimental e cobre o nucleo Transformer/DiT do F5-TTS. Ele nao e, sozinho, uma pipeline completa texto-para-WAV.

A inferencia completa ainda precisa de etapas ao redor:

- preprocessamento de texto;
- condicionamento por audio de referencia;
- loop de difusao/sampling;
- vocoder;
- escrita do WAV.

Por isso o pacote final preserva os arquivos originais para permitir inferencia Python completa com a qualidade original.

## Revisao 2026-06-16: pacote ONNX/Lite testavel

O empacotador foi atualizado para nao apresentar o `f5_tts_transformer_core.onnx` como pipeline TTS completo. A conclusao tecnica permanece: com o F5-TTS atual, um unico ONNX de alto nivel `text/text_ids -> waveform` nao e viavel neste empacotador, porque a inferencia completa depende de:

- tokenizer/preprocessamento em Python;
- condicionamento por audio e texto de referencia;
- loop iterativo de flow matching/sampling;
- vocoder `vocos`;
- pos-processamento e escrita WAV.

O pacote novo passa a ser um pipeline parcial documentado:

- `onnx/f5_tts_transformer_core.onnx`: nucleo DiT/Transformer exportado e validado com `onnxruntime`;
- `model/vocab.txt`: vocabulario usado pela voz;
- `model/<checkpoint>`: checkpoint principal da voz;
- `reference/referencia_voz.wav`: audio de referencia;
- `reference/reference_text.txt`: texto exato quando encontrado; se ausente, o arquivo registra que o F5-TTS tentara transcricao automatica;
- `manifest.json`: contrato do pacote, runtime necessario e limitacoes;
- `onnx_export_report.json`: inputs/outputs com nomes, shapes e tipos, teste CPU, arquivos gerados e limitacoes;
- `package_metadata.json`: metadados de origem/destino;
- `scripts/test_package_cpu.py`: valida o ONNX com `onnxruntime` e gera WAV em CPU usando `f5-tts` + `vocos`.

O teste CPU agora roda por padrao antes do upload. Comando reproduzivel dentro do pacote:

```bash
python scripts/test_package_cpu.py \
  --text "Boa noite Warllem, este é um teste do modo lite em CPU." \
  --output-wav test_outputs/voz_noslen_lite_cpu.wav \
  --nfe-step 4 \
  --speed 1.0
```

Se esse teste falhar, o script interrompe a publicacao por padrao. `--skip-cpu-test` e `--allow-failed-cpu-test` ficam disponiveis apenas para diagnostico, nao para pacote final validado.

## Como verificar no Kaggle

Ao rodar a celula 5, o log deve mostrar:

```text
Voz_Noslen ONNX packager versao: 2026.06.15.6
```

Se essa linha nao aparecer, o notebook/script antigo ainda esta sendo executado. Nesse caso, rode novamente desde a celula 1 ou reimporte o notebook atualizado.
