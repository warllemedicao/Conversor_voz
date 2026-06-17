# Voz_Noslen F5-TTS ONNX (Modo Lite) no Kaggle

Use `voz_noslen_f5_tts_onnx_kaggle.ipynb` em um notebook Kaggle com Internet ativada.

## Objetivo atual

Criar um pacote ONNX (Modo Lite) da voz neural treinada `Voz_Noslen`, otimizado para execução no Cloud Run.

O pacote final contem somente o runtime minimo:

```text
/onnx_package_name/
├── onnx/
│   └── f5_tts_transformer_core.onnx  <-- (Gerado após a conversão)
├── model/
│   ├── model_2000.pt                <-- (Checkpoint original PyTorch)
│   └── vocab.txt                    <-- (Dicionário de caracteres/tokens)
└── reference/
    └── referencia_voz.wav           <-- (Áudio de referência para clonagem)
```

## Politica de qualidade

Para preservar a qualidade da voz treinada:

- o ONNX principal usa a precisao original do checkpoint;
- o checkpoint treinado (`model_2000.pt`), `vocab.txt` e audio de referencia são mantidos;
- o arquivo .pt é necessário pois o motor Lite o utiliza para inicializar os metadados da arquitetura.

## Contrato do ONNX (Modo Lite)

* **Nome**: f5_tts_transformer_core.onnx
* **Opset**: 17
* **Entradas (Inputs)**: text_ids, text_lengths, ref_text_ids, ref_text_lengths, speed, n_steps.
* **Saída (Output)**: audio (com eixos dinâmicos para o comprimento do áudio).

## Como rodar

1. Abra `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` no Kaggle.
2. Ative Internet.
3. Se quiser upload para Hugging Face, adicione um Secret chamado `HF_TOKEN`.
4. Execute **Run All**.

Por padrao, o notebook:

- baixa somente arquivos essenciais do bucket;
- exporta o ONNX no formato Lite;
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
