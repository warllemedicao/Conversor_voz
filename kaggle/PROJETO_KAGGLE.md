# Projeto Super Voz no Kaggle

Este documento descreve como ficou a versao Kaggle do projeto, de onde os arquivos vem, quais arquivos do treinamento devem ser usados e como o notebook deve funcionar.

## Objetivo

O objetivo da pasta `kaggle` e permitir rodar a voz neural Super Voz dentro do Kaggle, usando GPU e baixando os arquivos do modelo diretamente do Hugging Face. A versao Kaggle nao depende mais do Google Drive usado no notebook Colab original.

O notebook principal e:

```text
kaggle/conversor_voz_kaggle.ipynb
```

Ele cria o modulo Python dentro de `/kaggle/working`, baixa o pacote `warllem/Super_voz`, detecta o melhor checkpoint do treino, escolhe o audio de referencia e carrega a voz. Depois disso, o usuario pode gerar audio de duas formas:

1. Por uma celula simples do notebook, que mostra player e link de download direto.
2. Pela interface Gradio opcional.

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

O download e seletivo. O repositorio do GitHub nao guarda os audios de referencia nem os pesos do modelo. Durante a execucao no Kaggle, o notebook baixa do Hugging Face apenas os padroes necessarios:

```text
model/**
docs/**
inference/**
tokenizer/**
data_reference/referencia_voz.wav
data_reference/*.txt
data_reference/*.csv
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

### 6. Carregamento da voz

Depois de instalar as dependencias, o notebook carrega a voz uma unica vez com:

```python
synthesizer = load_synthesizer(download=False)
```

Esse objeto `synthesizer` fica disponivel nas proximas celulas. Assim, nao e preciso baixar nem carregar o modelo de novo a cada texto.

### 7. Geracao simples com download direto

A forma recomendada no Kaggle e usar a celula:

```python
texto = 'Digite aqui o texto que voce quer transformar em audio.'

audio_path = synthesize_for_notebook(synthesizer, texto)
audio_path
```

Ao executar essa celula, o programa:

1. Gera o audio.
2. Salva o WAV em `/kaggle/working/audios_gerados`.
3. Mostra um player para ouvir o audio no output da celula.
4. Mostra um link `Download do WAV` no output da celula.

Os arquivos gerados ficam com nomes sequenciais:

```text
/kaggle/working/audios_gerados/audio_0001.wav
/kaggle/working/audios_gerados/audio_0002.wav
/kaggle/working/audios_gerados/audio_0003.wav
```

Para gerar outro audio, basta trocar o valor de `texto` e executar a mesma celula de novo.

Essa opcao e melhor para Kaggle porque nao depende de procurar pastas na lateral do ambiente.

### 8. Interface Gradio opcional

Tambem existe uma celula opcional para abrir Gradio:

```python
demo = create_gradio_app(synthesizer)
demo.launch(share=True, debug=True)
```

O usuario digita uma frase, pressiona Enter, e o sistema gera um arquivo WAV em:

```text
/kaggle/working/audios_gerados
```

Na interface Gradio, o audio aparece como player e como arquivo para download. Porem, enquanto essa celula estiver rodando, ela segura a execucao do notebook. Para executar outras celulas depois, e preciso parar a celula do Gradio.

## Arquivos criados na pasta kaggle

A pasta `kaggle` ficou com:

```text
README_kaggle.md
PROJETO_KAGGLE.md
conversor_voz_kaggle.ipynb
conversor_voz_kaggle.py
conversor_voz_requirements_kaggle.txt
```

Nao ha arquivos Colab dentro da pasta `kaggle`, para evitar conflito de notebook, modulo ou requirements. Os pesos e audios tambem nao ficam versionados dentro do GitHub. Eles sao baixados pelo notebook dentro do Kaggle, a partir do Hugging Face.

Os arquivos usados pelo fluxo Kaggle sao:

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

8. Na celula de geracao simples, altere:

```python
texto = 'Digite aqui o texto que voce quer transformar em audio.'
```

9. Execute a celula para ouvir e baixar o WAV direto pelo output.
10. Use a interface Gradio apenas se quiser a caixa de texto interativa.

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
Modo recomendado: celula synthesize_for_notebook com player e link de download
```

Se o Hugging Face mudar a estrutura no futuro, o codigo ainda tenta detectar checkpoints, configs, logs e audios por busca automatica. Mas para a estrutura atual, os arquivos acima sao os corretos.
