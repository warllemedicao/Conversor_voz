# Relatorio Voz_Noslen F5-TTS ONNX - Modo Turbo

Este documento registra o estado atual do notebook Kaggle e do packager ONNX para a voz neural `Voz_Noslen`.

## Objetivo

Gerar, no Kaggle, um pacote Turbo para execução em backend Python com ONNX Runtime. O pacote exporta apenas o nucleo Transformer/DiT do F5-TTS; o loop de difusao, vocoder e demais controles de inferencia permanecem no backend.

## Versao atual

```text
PACKAGER_VERSION=2026.06.19.turbo.v5
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
    - `x`: tensor latente `float32`, shape `[1, 128, 100]`.
    - `cond`: condicionamento mel `float32`, shape `[1, 128, 100]`.
    - `text_ids`: IDs de texto `int64`, shape `[1, text_len]`.
    - `text_lengths`: comprimento de texto `int64`, shape `[1]`.
    - `time_steps`: tempo da difusao `float32`, shape `[1]`.
* **Saida:**
    - `dx`: velocidade prevista pelo Transformer, shape `[1, 128, 100]`.

## Correcao aplicada em 2026-06-19

O erro informado no Kaggle nao indicava falha do checkpoint em si; a falha ocorreu na etapa de exportacao ONNX. O wrapper chamava o DiT com `text_lengths` como quinto argumento posicional, mas esse parametro era interpretado pelo F5-TTS como controle/máscara de audio. Isso fazia `audio_mask` chegar como tensor 1D e quebrava em `audio_mask.sum(dim=1)`.

A chamada foi corrigida para argumentos nomeados:

```python
self.transformer(x=x, cond=cond, text=text_ids, time=time_steps)
```

Para preservar o contrato ONNX, `text_lengths` continua como entrada do grafo por uma ancora dinamica (`x + length_anchor - length_anchor`), sem ser repassado como argumento opcional do DiT.

## Correcao complementar em 2026-06-19

A execucao seguinte no Kaggle ainda falhou durante `torch.onnx.export`, depois do carregamento de `model_last.pt`, com a mesma mensagem curta `Dimension out of range`. A versao `2026.06.19.turbo.v4` tornou o wrapper mais explicito:

- cria `audio_mask` 2D com shape `[batch, duration]` a partir de `x`;
- passa `mask=audio_mask` somente quando a assinatura instalada do `transformer.forward` suporta esse argumento;
- passa `cache=False` somente quando a assinatura suporta esse argumento;
- registra no log a assinatura real de `transformer.forward`;
- usa `LOGGER.exception` para imprimir traceback completo em novas falhas.

Com isso, se o exportador entrar no caminho de `audio_mask.sum(dim=1)`, a máscara enviada pelo wrapper tem duas dimensoes e satisfaz o contrato esperado pelo DiT.

## Correcao complementar em 2026-06-19 - validacao ONNX Runtime

A execucao manual da celula 4 gerou o ONNX, mas a validacao falhou no ONNX Runtime:

```text
Concat node /transformer/input_embed/Concat
Non concat axis dimensions must match: Axis 1 has mismatched dimensions of 128 and 16
```

A causa foi o tracer legado do PyTorch especializar o comprimento interno do `TextEmbedding` em `128` frames por causa de `seq_len.max().item()`, enquanto o smoke test ainda alimentava `x` e `cond` com `16` frames. A versao `2026.06.19.turbo.v5` assume explicitamente `TURBO_DURATION=128`, remove eixo dinâmico de `x`, `cond` e `dx`, atualiza o metadata e roda o smoke test com `[1, 128, 100]`.

Essa versao gera um ONNX valido para chamadas Turbo de 128 frames. O backend deve dividir ou preencher os tensores nesse tamanho antes de chamar o grafo.

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
