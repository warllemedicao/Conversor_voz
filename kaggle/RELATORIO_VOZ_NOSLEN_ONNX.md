# Relatorio Voz_Noslen F5-TTS ONNX - Modo Turbo

Este documento registra o estado atual do notebook Kaggle e do packager ONNX para a voz neural `Voz_Noslen`.

## Objetivo

Gerar, no Kaggle, um pacote Turbo para execução em backend Python com ONNX Runtime. O pacote exporta apenas o nucleo Transformer/DiT do F5-TTS; o loop de difusao, vocoder e demais controles de inferencia permanecem no backend.

## Versao atual

```text
PACKAGER_VERSION=2026.06.19.turbo.v3
```

## Arquivos sincronizados

- `kaggle/f5_tts_onnx_packager_kaggle.py`
- `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb`
- `kaggle/README_kaggle.md`
- `kaggle/CORRECOES_ERROS.md`
- `kaggle/RELATORIO_VOZ_NOSLEN_ONNX.md`

## Formato final do pacote

```text
turbo_staging_area/
├── onnx/
│   └── f5_tts_transformer_core.onnx
├── model/
│   └── vocab.txt
├── reference/
│   └── referencia_voz.wav
├── manifest.json
├── metadata.json
└── validation.json
```

## Contrato ONNX Turbo

* **Arquivo:** `f5_tts_transformer_core.onnx`
* **Opset:** 17
* **Entradas:**
    - `x`: tensor latente `float32`, shape `[1, duration, 100]`.
    - `cond`: condicionamento mel `float32`, shape `[1, duration, 100]`.
    - `text_ids`: IDs de texto `int64`, shape `[1, text_len]`.
    - `text_lengths`: comprimento de texto `int64`, shape `[1]`.
    - `time_steps`: tempo da difusao `float32`, shape `[1]`.
* **Saida:**
    - `dx`: velocidade prevista pelo Transformer, shape `[1, duration, 100]`.

## Correcao aplicada em 2026-06-19

O erro informado no Kaggle nao indicava falha do checkpoint em si; a falha ocorreu na etapa de exportacao ONNX. O wrapper chamava o DiT com `text_lengths` como quinto argumento posicional, mas esse parametro era interpretado pelo F5-TTS como controle/máscara de audio. Isso fazia `audio_mask` chegar como tensor 1D e quebrava em `audio_mask.sum(dim=1)`.

A chamada foi corrigida para argumentos nomeados:

```python
self.transformer(x=x, cond=cond, text=text_ids, time=time_steps)
```

Para preservar o contrato ONNX, `text_lengths` continua como entrada do grafo por uma ancora dinamica (`x + length_anchor - length_anchor`), sem ser repassado como argumento opcional do DiT.

## Criterio para upload

O upload so deve ser considerado seguro quando:

```text
validation.json -> "status": "verified"
```

Se a exportacao ONNX falhar ou o smoke test do ONNX Runtime nao passar, o pacote nao deve ser enviado como versao pronta.

## Validacoes locais

Validacoes possiveis sem baixar modelos grandes:

```bash
python -m py_compile kaggle/f5_tts_onnx_packager_kaggle.py
python -m json.tool kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb
```

O teste real de exportacao depende do ambiente Kaggle com Internet, `f5-tts`, checkpoint, vocabulario e audio de referencia baixados.
