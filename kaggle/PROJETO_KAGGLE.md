# Projeto Super Voz no Kaggle

Este documento descreve como ficou a versao Kaggle do projeto, de onde os arquivos vem, quais arquivos do treinamento devem ser usados e como o notebook deve funcionar.

## Objetivo

O objetivo da pasta `kaggle` e permitir rodar a voz neural Super Voz dentro do Kaggle, usando GPU e baixando os arquivos do modelo diretamente do Hugging Face. A versao Kaggle nao depende mais do Google Drive usado no notebook Colab original.

O notebook principal e:

```text
kaggle/conversor_voz_kaggle.ipynb
```

Ele cria o modulo Python dentro de `/kaggle/working`, baixa o pacote `warllem/Super_voz`, detecta o melhor checkpoint do treino, escolhe o audio de referencia e abre uma interface Gradio para gerar arquivos WAV.

## Origem dos arquivos

Os arquivos do modelo vem do Hugging Face:

```text
https://huggingface.co/warllem/Super_voz
```

Repositorio usado pelo notebook:

```text
warllem/Super_voz
```

O notebook baixa esse repositorio com:

```python
huggingface_hub.snapshot_download(...)
```

O destino no Kaggle e:

```text
/kaggle/working/Super_voz
```

Para acessar o Hugging Face, o notebook procura um token salvo nos Kaggle Secrets com um destes nomes:

```text
HF_TOKEN
HUGGINGFACE_TOKEN
HUGGING_FACE_HUB_TOKEN
```

O nome recomendado e:

```text
HF_TOKEN
```

## Arquivos corretos do treinamento

Foi feita uma leitura da estrutura real do repositorio `warllem/Super_voz`. Os arquivos importantes encontrados foram:

```text
model/config.yml
model/best_metric.txt
model/best_model.pth
model/latest_checkpoint.pth
model/latest_checkpoint.txt
model/Utils/ASR/epoch_00080.pth
model/Utils/JDC/bst.t7
model/Utils/PLBERT/step_1000000.t7
data_reference/referencia_voz.wav
docs/train.log
```

Esses sao os arquivos que a versao Kaggle deve usar para inferencia.

## Melhor checkpoint do treino

O arquivo mais importante para decidir qual checkpoint usar e:

```text
model/best_metric.txt
```

Ele informa:

```text
source_checkpoint=epoch_2nd_00045.pth
epoch=45
validation_loss=0.268
```

Isso significa que o melhor resultado do treinamento veio do epoch 45, com `validation_loss=0.268`.

No pacote final do Hugging Face, esse melhor checkpoint esta consolidado como:

```text
model/best_model.pth
```

Por isso, o programa foi configurado para usar primeiro:

```text
model/best_model.pth
```

Se esse arquivo nao existir, o programa tenta ler os logs de treino para encontrar o epoch com menor `Validation loss`. Se ainda assim nao conseguir, ele usa o checkpoint mais recente:

```text
model/latest_checkpoint.pth
```

## Logs de treinamento

O log final correto do pacote esta em:

```text
docs/train.log
```

Tambem existe outro log historico em:

```text
StyleTTS2/Models/super_Voz/train.log
```

O codigo foi ajustado para priorizar `docs/train.log`, porque ele representa o pacote final exportado. Isso evita que o programa escolha um epoch errado com base em logs antigos.

Quando `best_metric.txt` existe, ele tem prioridade sobre os logs. Isso e importante porque o pacote ja informa explicitamente qual checkpoint foi escolhido como melhor.

## Audio de referencia

O audio de referencia principal encontrado no pacote foi:

```text
data_reference/referencia_voz.wav
```

O programa possui uma funcao para tentar escolher o melhor audio de referencia automaticamente:

1. Procura nos logs linhas que tenham um arquivo `.wav` junto com alguma metrica de analise, como `score`, `similarity`, `mos`, `quality`, `loss`, `wer` ou `cer`.
2. Se encontrar uma linha com audio e metrica, escolhe o melhor audio pela metrica.
3. Se nao encontrar audio analisado nos logs, usa:

```text
data_reference/referencia_voz.wav
```

4. Se esse arquivo nao existir, tenta procurar um WAV em `val_list.txt`.
5. Se ainda nao encontrar, usa o primeiro WAV disponivel em `data_reference`.

No pacote atual, a escolha esperada e:

```text
data_reference/referencia_voz.wav
```

## Motor de voz detectado

O pacote e detectado como:

```text
styletts2
```

A deteccao acontece porque o arquivo `model/config.yml` contem marcadores esperados do StyleTTS2, como:

```text
ASR_config
PLBERT_dir
model_params
preprocess_params
```

Entao a inferencia e carregada com:

```text
model/best_model.pth
model/config.yml
```

E usa os auxiliares:

```text
model/Utils/ASR/epoch_00080.pth
model/Utils/JDC/bst.t7
model/Utils/PLBERT/step_1000000.t7
```

## Como o notebook funciona

O notebook `conversor_voz_kaggle.ipynb` executa as etapas abaixo.

### 1. Conferencia de GPU e token

Verifica se a GPU esta disponivel no Kaggle e procura o token do Hugging Face nos Kaggle Secrets.

### 2. Instalacao minima

Instala apenas o necessario para baixar e detectar o pacote:

```text
huggingface_hub
hf_xet
pyyaml
espeak-ng
ffmpeg
```

### 3. Criacao do modulo

O notebook grava o codigo de `conversor_voz_kaggle.py` em:

```text
/kaggle/working/conversor_voz_kaggle.py
```

Isso permite rodar o notebook de forma independente no Kaggle.

### 4. Download do Hugging Face

Baixa o repositorio:

```text
warllem/Super_voz
```

para:

```text
/kaggle/working/Super_voz
```

Depois imprime os arquivos encontrados e detecta:

```text
Engine: styletts2
Modelo: /kaggle/working/Super_voz/model/best_model.pth
Config: /kaggle/working/Super_voz/model/config.yml
Audio referencia: /kaggle/working/Super_voz/data_reference/referencia_voz.wav
```

### 5. Instalacao do motor detectado

Depois que o motor e detectado, instala somente as dependencias necessarias para ele.

Para `styletts2`, instala:

```text
styletts2
gradio
pydub
soundfile
pyyaml
numpy==1.26.4
```

O `numpy==1.26.4` e reinstalado para reduzir erro binario comum em ambientes com pacotes cientificos, como:

```text
numpy.dtype size changed
```

### 6. Interface de uso

Por fim, carrega o modelo e abre uma interface Gradio.

O usuario digita uma frase, pressiona Enter, e o sistema gera um arquivo WAV em:

```text
/kaggle/working/audios_gerados
```

## Arquivos criados na pasta kaggle

A pasta `kaggle` ficou com:

```text
README.md
README_kaggle.md
PROJETO_KAGGLE.md
conversor_voz_kaggle.ipynb
conversor_voz_kaggle.py
conversor_voz_requirements_kaggle.txt
conversor_voz_colab.py
conversor_voz_one_click_colab.ipynb
conversor_voz_requirements_colab.txt
audio_warllem_ref_texto.wav
audio_warllem_ref_fonemas.wav
audio_warllem_ref_fonemas_ptbr.wav
audio_warllem_voz_neural.wav
```

Os arquivos Colab e audios locais foram copiados para a pasta `kaggle` como clone/base do projeto original. Os arquivos realmente usados pelo fluxo Kaggle novo sao:

```text
conversor_voz_kaggle.ipynb
conversor_voz_kaggle.py
conversor_voz_requirements_kaggle.txt
README_kaggle.md
PROJETO_KAGGLE.md
```

## Como deve ser usado no Kaggle

1. Abra o Kaggle.
2. Crie um notebook com GPU ligada.
3. Configure o secret `HF_TOKEN` com o token do Hugging Face.
4. Suba ou copie o notebook `conversor_voz_kaggle.ipynb`.
5. Execute as celulas em ordem.
6. Aguarde o download do repositorio `warllem/Super_voz`.
7. Confirme que a saida mostra:

```text
Engine detectada: styletts2
Modelo: /kaggle/working/Super_voz/model/best_model.pth
Config: /kaggle/working/Super_voz/model/config.yml
Audio referencia: /kaggle/working/Super_voz/data_reference/referencia_voz.wav
```

8. Use a interface Gradio para digitar frases e gerar WAV.

## Comportamento esperado

O comportamento correto esperado e:

```text
Origem dos arquivos: Hugging Face warllem/Super_voz
Pasta local no Kaggle: /kaggle/working/Super_voz
Motor: StyleTTS2
Checkpoint: model/best_model.pth
Config: model/config.yml
Audio de referencia: data_reference/referencia_voz.wav
Saida dos audios: /kaggle/working/audios_gerados
```

Se o Hugging Face mudar a estrutura no futuro, o codigo ainda tenta detectar checkpoints, configs, logs e audios por busca automatica. Mas para a estrutura atual, os arquivos acima sao os corretos.
