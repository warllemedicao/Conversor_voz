# Projeto Super Voz F5-TTS no Kaggle

Este projeto executa exclusivamente uma voz F5-TTS treinada no Hugging Face.

## Historico

### 2026-06-10

- Reescrito o carregador Kaggle para trabalhar apenas com F5-TTS.
- Implementada listagem recursiva do Hugging Face com metadados de tamanho/LFS.
- Implementada politica explicita de selecao de checkpoint pelo manifesto.
- Implementadas validacoes de manifesto, vocabulario, checkpoint, referencia de audio, arquitetura F5TTS_v1_Base e vocoder Vocos.
- Registrados artefatos presentes e ausentes para inferencia e para retomada/reproducao de treinamento.
- Removida a tolerancia a fallback silencioso quando a API do Hugging Face falha; o erro original agora e registrado com traceback completo.

## Fluxo atual

1. Prepara ambiente Kaggle sem reinstalar indiscriminadamente NumPy/SciPy/Pandas.
2. Instala `huggingface_hub`, `hf-xet`, `f5-tts`, `torch`, `torchaudio`, `soundfile` e auxiliares.
3. Lista recursivamente `warllem/Super_voz`.
4. Baixa e valida `voices/v_minha_voz_f5_tts_ptbr/manifest.json`.
5. Resolve o checkpoint recomendado pelo manifesto.
6. Baixa somente checkpoint escolhido, vocabularios e referencia.
7. Inspeciona checkpoint e vocabulario.
8. Carrega F5-TTS, Vocos e gera WAV.

## Politica de selecao do checkpoint

Ordem:

1. `voice_checkpoint` do manifesto.
2. `inference_checkpoint` ou `final_checkpoint`, se existirem.
3. `latest_checkpoint` do manifesto.
4. Fallback F5 validado por existencia remota, sem usar caminhos de estruturas antigas.

No estado auditado em 2026-06-10, o manifesto aponta para:

```text
voices/v_minha_voz_f5_tts_ptbr/model/model_2000.pt
```

Esse arquivo tem aproximadamente 5.39 GB e compartilha o mesmo objeto LFS com:

```text
voices/v_minha_voz_f5_tts_ptbr/model/latest_checkpoint.pt
```

## Arquivos obrigatorios para inferencia

Presentes:

```text
voices/v_minha_voz_f5_tts_ptbr/manifest.json
voices/v_minha_voz_f5_tts_ptbr/model/model_2000.pt
voices/v_minha_voz_f5_tts_ptbr/model/latest_checkpoint.pt
voices/v_minha_voz_f5_tts_ptbr/model/model_last.pt
voices/v_minha_voz_f5_tts_ptbr/model/vocab.txt
voices/v_minha_voz_f5_tts_ptbr/data_reference/referencia_voz.wav
libraries/f5_tts_ptbr_tharyck/setting.json
libraries/f5_tts_ptbr_tharyck/vocab.txt
libraries/f5_tts_ptbr_tharyck/model_last.safetensors
```

Ausente:

```text
voices/v_minha_voz_f5_tts_ptbr/data_reference/referencia_voz.txt
```

Como a transcricao exata da referencia nao esta publicada, o carregador registra o fato no diagnostico e permite que o F5-TTS use ASR automaticamente.

## Arquitetura recuperada

O manifesto informa:

```text
architecture: F5-TTS
exp_name: F5TTS_v1_Base
tokenizer: char
```

O pacote nao publica um `model_config.json` completo. O carregador recupera a configuracao padrao `F5TTS_v1_Base` do runtime F5-TTS:

```text
backbone: DiT
dim: 1024
depth: 22
heads: 16
ff_mult: 2
text_dim: 512
text_mask_padding: true
conv_layers: 4
target_sample_rate: 24000
n_mel_channels: 100
hop_length: 256
win_length: 1024
n_fft: 1024
mel_spec_type: vocos
```

## Arquivos para retomar ou reproduzir treinamento

Presentes parcialmente:

- checkpoint da voz;
- checkpoint-base da biblioteca PT-BR;
- `setting.json` com parametros gerais de treinamento;
- `duration.json`;
- vocabulario.

Ausentes ou nao publicados:

- estado de otimizador;
- scheduler;
- scaler AMP;
- seed;
- commit exato do F5-TTS usado no fine-tuning;
- versoes completas das dependencias;
- dataset da voz personalizada;
- `metadata.csv`, `raw.arrow` ou equivalente;
- divisao train/validation;
- logs e metricas completas.

Esses itens nao bloqueiam inferencia, mas bloqueiam reproducao fiel ou retomada completa do treinamento.
