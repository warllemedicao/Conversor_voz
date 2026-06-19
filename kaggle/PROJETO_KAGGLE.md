# Projeto Kaggle F5-TTS ONNX Lite CPU

## Objetivo

Gerar no Kaggle um pacote utilizavel pelo backend Lite CPU sem degradar a voz base. O formato alvo e ONNX para o nucleo F5-TTS/DiT, mantendo preprocessamento, loop de difusao, Vocos e escrita de WAV em Python.

## Fluxo Atual

1. O notebook baixa os ativos do Hugging Face Storage Bucket `https://huggingface.co/buckets/warllem/Voz_Noslen`.
2. A arvore esperada e `voices/v_minha_voz_f5_tts_ptbr`.
3. O packager procura manifesto fonte quando houver.
4. O checkpoint e escolhido nesta ordem: manifesto, `model_last.pt`, maior `model_*.pt`.
5. `vocab.txt` e `data_reference/referencia_voz.wav` sao validados.
6. O F5-TTS v1 Base e carregado em CPU com `mel_spec_type="vocos"`.
7. O ONNX exporta somente o passo do DiT core.
8. `onnxruntime` CPU abre o grafo, valida inputs/outputs e mede o tempo de uma chamada dummy do core.
9. Um smoke test separado gera `validation/full_pipeline_smoke.wav` pelo pipeline Python F5-TTS + Vocos.
10. O zip/upload so acontece quando `validation.json` fica com `status: verified`.

## Contrato ONNX

Arquivo: `onnx/f5_tts_transformer_core.onnx`

Entradas:

- `x`: `float32`, shape `[1, 128, 100]`
- `cond`: `float32`, shape `[1, 128, 100]`
- `text_ids`: `int64`, shape `[1, text_len]`
- `text_lengths`: `int64`, shape `[1]`
- `time_steps`: `float32`, shape `[1]`

Saida:

- `dx`: `float32`, shape `[1, 128, 100]`

Este contrato nao e text-to-audio end-to-end. O Lite CPU precisa manter em Python a preparacao dos tensores, loop de difusao/ODE, vocoder Vocos e escrita do WAV a 24000 Hz.

## Politica de Qualidade

- Sem quantizacao agressiva.
- Sem pruning.
- Sem troca de vocoder.
- Sem reducao agressiva de `nfe_step` para mascarar custo de CPU.
- Sem alteracao da identidade da voz.

## Arquivos

- `f5_tts_onnx_packager_kaggle.py`: packager canonico.
- `voz_noslen_f5_tts_onnx_kaggle.ipynb`: notebook canonico.
- `conversor_voz_kaggle.py`: alias para compatibilidade.
- `README_kaggle.md`: guia operacional.
- `RELATORIO_VOZ_NOSLEN_ONNX.md`: historico tecnico de correcoes.
