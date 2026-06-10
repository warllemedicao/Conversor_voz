# Conversa Sobre Voz Neural Portavel

## Historico de Atualizacoes

### 2026-06-10 - Mudanca para F5-TTS no Kaggle

O projeto Kaggle foi atualizado para usar exclusivamente a voz F5-TTS publicada em:

```txt
https://huggingface.co/warllem/Super_voz
```

O pacote de voz atual fica em:

```txt
voices/v_minha_voz_f5_tts_ptbr/
```

O manifesto seleciona o checkpoint:

```txt
voices/v_minha_voz_f5_tts_ptbr/model/model_2000.pt
```

O fluxo Kaggle agora baixa apenas manifesto, vocabularios, referencia e o checkpoint escolhido, valida os artefatos e gera o WAV final em:

```txt
/kaggle/working/resultado_voz.wav
```

As secoes abaixo preservam o historico anterior de testes com StyleTTS2/Colab.

## Objetivo

Criar uma forma de usar a voz neural masculina em portugues do Brasil, treinada com StyleTTS2, em diferentes produtos:

- Extensao Chrome de leitura de texto.
- Futuramente uma skill da Alexa.
- Possivelmente outros apps/sites.

O objetivo nao e usar a voz do Microsoft Edge ou da Alexa. Essas vozes foram citadas apenas como exemplo de um modelo empacotado/portavel.

## Estado Atual Do Modelo

Foram analisados arquivos vindos do Google Drive contendo um projeto StyleTTS2:

- `Models/super_Voz/epoch_2nd_00019.pth`
- `Models/super_Voz/epoch_2nd_00024.pth`
- `Models/super_Voz/epoch_2nd_00029.pth`
- `Configs/config_super_voz.yml`
- `Utils/ASR/epoch_00080.pth`
- `Utils/JDC/bst.t7`
- `Utils/PLBERT/step_1000000.t7`

O checkpoint mais recente usado nos testes foi:

```txt
epoch_2nd_00029.pth
```

Tambem foi usado um audio de referencia masculino enviado depois:

```txt
/tmp/warllem_ref_24k_mono.wav
```

Esse audio foi convertido para WAV mono 24 kHz para uso como referencia de estilo.

## Audios Gerados

Foram gerados alguns audios de teste:

```txt
audio_warllem_voz_neural.wav
audio_warllem_ref_texto.wav
audio_warllem_ref_fonemas.wav
audio_warllem_ref_fonemas_ptbr.wav
```

O melhor resultado foi:

```txt
audio_warllem_ref_fonemas.wav
```

Depois foi gerada uma versao com fonemas mais proximos de portugues do Brasil:

```txt
audio_warllem_ref_fonemas_ptbr.wav
```

## Problemas Encontrados

O pacote `styletts2` do PyPI usa `gruut` como fonemizador padrao. No teste, ele converteu a frase em fonemas de ingles, o que causou pronuncia errada.

Exemplo do comportamento ruim:

```txt
pˈɑɹəbənz wˈɚləm ...
```

Para melhorar, foi usado um fonemizador manual que passa fonemas portugueses diretamente para o modelo.

Tambem foram encontrados bugs/limites no pacote `styletts2==0.1.6`:

- Ele tentava baixar o F0 padrao mesmo existindo `Utils/JDC/bst.t7`.
- O caminho do PLBERT precisou ser passado como `Path`.
- Foi necessario ajustar caches para `/tmp`:
  - `MPLCONFIGDIR`
  - `HF_HOME`
  - `TRANSFORMERS_CACHE`
  - `NLTK_DATA`
  - `NUMBA_CACHE_DIR`

## Integracao Com A Extensao Chrome

A extensao atual usa `speechSynthesis` no `content.js`.

Esse caminho nao permite importar uma voz neural customizada treinada em StyleTTS2. O navegador so lista vozes instaladas/suportadas pelo sistema ou pelo proprio browser.

A integracao mais direta seria:

```txt
Extensao Chrome
  -> envia texto
Runtime local/API
  -> gera WAV/MP3 com a voz neural
Extensao Chrome
  -> toca o audio retornado
```

Mas foi levantada uma preocupacao correta: se isso for feito como servidor pesado ou carregando modelo a cada frase, a extensao ficara lenta.

## Esclarecimento Sobre Edge E Alexa

Nao existe um formato publico para converter diretamente o checkpoint StyleTTS2 em uma voz nativa do Microsoft Edge, Windows ou `speechSynthesis`.

Tambem nao da para instalar uma voz neural customizada dentro da Alexa como se fosse uma voz nativa da Amazon.

Para Alexa Skill, o caminho e a skill chamar um motor TTS proprio e tocar audio gerado por ele via URL HTTPS em formato aceito pela Alexa.

## Direcao Correta

Criar um pacote portavel da voz, por exemplo:

```txt
warllem_voice/
  checkpoint.pth
  config.yml
  asr.pth
  f0.t7
  plbert.t7
  style.npy
  phonemizer_rules.json
  runtime
```

Ou um arquivo compactado unico:

```txt
warllem_voice_v1.wvoice
```

Por dentro, esse pacote teria os pesos, configs, fonemizador e embedding fixo da voz.

## Plano Recomendado

1. Criar um runtime Python estavel.

   Exemplo:

   ```bash
   python voice_runtime/synthesize.py \
     "Parabens warllem, conseguimos. A sua voz neural esta otima" \
     --out audio.wav
   ```

2. Calcular e salvar o style embedding uma vez.

   Em vez de depender sempre do audio de referencia, salvar:

   ```txt
   warllem_style.npy
   ```

3. Criar um pacote da voz.

   Exemplo:

   ```txt
   voice_runtime/
     synthesize.py
     voice_package.py
     phonemizer_ptbr.py
     warllem_voice.json
   ```

4. Usar esse runtime em diferentes ambientes:

   ```txt
   Chrome Extension -> Native Messaging ou API local
   Alexa Skill      -> API HTTPS que gera/serve MP3
   App/Site         -> API HTTP
   Desktop          -> CLI local
   ```

5. Depois avaliar exportacao para ONNX.

   StyleTTS2 nao vira facilmente um unico `.onnx`, porque a inferencia envolve:

   ```txt
   texto -> fonemas -> tokens -> BERT -> diffusion -> duracao -> F0 -> decoder -> audio
   ```

   O caminho realista seria exportar submodelos:

   - `bert`
   - `bert_encoder`
   - `predictor`
   - `text_encoder`
   - `style_encoder`
   - `predictor_encoder`
   - `decoder`

   E manter a logica de inferencia em um runtime.

## Conclusao

O melhor proximo passo nao e tentar criar uma voz nativa do Edge ou da Alexa. O melhor caminho e criar um pacote/runtime da voz Warllem:

```txt
texto -> normalizador PT-BR -> fonemizador PT-BR -> StyleTTS2 empacotado -> WAV/MP3
```

Depois esse runtime pode ser usado pela extensao Chrome, pela Alexa Skill e por outros sistemas.
