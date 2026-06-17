# Voz_Noslen F5-TTS ONNX Turbo no Kaggle

Use `voz_noslen_f5_tts_onnx_kaggle.ipynb` em um notebook Kaggle com Internet ativada.

## Objetivo atual

Criar um pacote ONNX Turbo da voz neural treinada `Voz_Noslen`, usando F5-TTS, sem carregar no pacote final a arvore antiga de treino.

O pacote final contem somente o runtime minimo:

```text
model/
reference/
onnx/
scripts/
manifest.json
package_metadata.json
onnx_export_report.json
```

A pasta antiga `f5_tts_original/` nao e mais publicada no pacote final.

## Politica de qualidade

Para preservar a qualidade da voz treinada:

- o ONNX principal usa a precisao original do checkpoint;
- a quantizacao INT8 fica desativada por padrao;
- o checkpoint treinado, `vocab.txt` e audio de referencia continuam sendo incluidos;
- a voz continua usando `F5TTS_v1_Base`, `vocos` e sample rate de 24000 Hz.

Se quiser gerar tambem INT8, defina:

```python
os.environ["F5_ONNX_QUANTIZE"] = "1"
```

Use INT8 apenas depois de validar a qualidade por escuta.

## Contrato do ONNX Turbo

O ONNX Turbo exportado encapsula:

- Transformer/DiT;
- loop Euler com `time_steps`;
- Vocos para gerar audio.

Entradas esperadas:

```text
x
cond
text
time_steps
mask
```

Saida:

```text
audio
```

Importante: ele nao e um grafo `texto cru -> WAV`. O backend ainda precisa preparar os IDs de texto, condicionamento da referencia, noise, mascara e passos de tempo antes de chamar o ONNX.

## Como rodar

1. Abra `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` no Kaggle.
2. Ative Internet.
3. Se quiser upload para Hugging Face, adicione um Secret chamado `HF_TOKEN`.
4. Execute **Run All**.

Por padrao, o notebook:

- baixa somente arquivos essenciais do bucket;
- exporta o ONNX Turbo em precisao original;
- roda teste CPU;
- envia para `warllem/Voz_Noslen_ONNX` se `HF_TOKEN` estiver disponivel.

## Origem e destino

Origem:

```text
https://huggingface.co/buckets/warllem/Voz_Noslen
```

Voz usada:

```text
voices/v_minha_voz_f5_tts_ptbr
```

Destino padrao:

```text
warllem/Voz_Noslen_ONNX
```

Pasta gerada:

```text
onnx_packages/turbo_<data_hora>
```

## Logs

O log completo fica em:

```text
/kaggle/working/voz_noslen_onnx_packager.log
```
