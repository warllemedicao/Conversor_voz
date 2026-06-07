# Projeto Super Voz no Kaggle

Este documento descreve a versao Kaggle do projeto e o comportamento esperado do notebook.

## Objetivo

A pasta `kaggle` contem somente os arquivos necessarios para rodar a Super Voz no Kaggle:

```text
README_kaggle.md
PROJETO_KAGGLE.md
conversor_voz_kaggle.ipynb
conversor_voz_kaggle.py
conversor_voz_requirements_kaggle.txt
```

Nao ha notebooks Colab, audios locais ou pesos versionados nessa pasta.

## Fluxo do notebook

O notebook `conversor_voz_kaggle.ipynb` deve rodar com `Run All` em um kernel limpo. Ele nao deve reiniciar o kernel intencionalmente.

As celulas fazem:

1. Conferem GPU e secret do Hugging Face, sem importar `torch`.
2. Instalam e validam dependencias antes de importar bibliotecas pesadas.
3. Criam `/kaggle/working/conversor_voz_kaggle.py`.
4. Baixam o pacote `warllem/Super_voz` do Hugging Face e detectam checkpoint/config/audio.
5. Validam dependencias e contexto.
6. Carregam o modelo.
7. Geram um WAV com player e link de download.
8. Opcionalmente abrem Gradio.

Se qualquer etapa falhar, a celula para e grava traceback em:

```text
/kaggle/working/super_voz_kaggle.log
```

## Origem dos arquivos

O modelo vem do Hugging Face:

```text
https://huggingface.co/warllem/Super_voz
```

Repositorio usado:

```text
warllem/Super_voz
```

O download vai para:

```text
/kaggle/working/Super_voz
```

O download e seletivo:

```text
model/**
docs/**
inference/**
tokenizer/**
data_reference/referencia_voz.wav
data_reference/*.txt
data_reference/*.csv
```

## Arquivos corretos do treinamento

Arquivos esperados no pacote:

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

`model/best_metric.txt` informa:

```text
source_checkpoint=epoch_2nd_00045.pth
epoch=45
validation_loss=0.268
```

O notebook usa primeiro:

```text
model/best_model.pth
```

Se esse arquivo faltar, o codigo tenta ler os logs e escolher o epoch com menor `Validation loss`. Se ainda assim falhar, usa `model/latest_checkpoint.pth`.

## Audio de referencia

O audio esperado e:

```text
data_reference/referencia_voz.wav
```

O codigo tambem procura nos logs algum `.wav` com metrica de analise. Se encontrar, usa o melhor audio pelo score. Se nao encontrar, usa `data_reference/referencia_voz.wav`.

## Dependencias

Os erros anteriores vieram de instalar ou recarregar NumPy/SciPy depois que o kernel ja tinha importado bibliotecas nativas. A versao atual evita isso:

- a celula 1 nao importa `torch`;
- a celula 2 instala dependencias antes de qualquer import pesado;
- a validacao de dependencias ocorre em subprocesso;
- o notebook nao usa `os._exit(0)` para reiniciar o kernel;
- o `styletts2==0.1.6` e instalado com `--no-deps`.

Versoes principais:

```text
numpy==1.26.4
scipy==1.12.0
pandas==2.2.2
styletts2==0.1.6
```

Avisos de `pip` sobre pacotes do ambiente global do Kaggle, como `google-cloud`, `bigquery`, `dask-cuda`, `jax` e `opencv`, nao significam uso de Google Drive ou Colab. O projeto usa Hugging Face.

## Saida

Os audios gerados ficam em:

```text
/kaggle/working/audios_gerados
```

A celula de geracao mostra player e link `Download do WAV`.
