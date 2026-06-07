# Super Voz no Kaggle

Use `conversor_voz_kaggle.ipynb` em um notebook Kaggle com GPU e internet ligadas.

## Como rodar

1. Configure o secret `HF_TOKEN` no Kaggle com seu token do Hugging Face.
2. Abra `conversor_voz_kaggle.ipynb`.
3. Clique em `Run All`.
4. Se tudo estiver correto, o notebook baixa o modelo, carrega a voz e gera um WAV na celula de geracao.

O fluxo atual nao reinicia o kernel de proposito. Se alguma celula falhar, a execucao para e o log completo fica em:

```text
/kaggle/working/super_voz_kaggle.log
```

## Origem dos arquivos

Os pesos, logs e audio de referencia vem do Hugging Face:

```text
warllem/Super_voz
```

O download seletivo traz apenas:

```text
model/**
docs/**
inference/**
tokenizer/**
data_reference/referencia_voz.wav
data_reference/*.txt
data_reference/*.csv
```

Nada disso fica salvo no GitHub; e baixado dentro do Kaggle em:

```text
/kaggle/working/Super_voz
```

## Arquivos usados

O pacote esperado contem:

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

O melhor checkpoint vem de `model/best_metric.txt`:

```text
source_checkpoint=epoch_2nd_00045.pth
epoch=45
validation_loss=0.268
```

No pacote final, ele e usado como:

```text
model/best_model.pth
```

## Dependencias

O notebook instala as dependencias antes de importar bibliotecas pesadas como `torch`, `numpy`, `scipy` ou `styletts2`. Isso evita o erro de recarregar NumPy/SciPy no mesmo processo.

As versoes principais fixadas sao:

```text
numpy==1.26.4
scipy==1.12.0
pandas==2.2.2
styletts2==0.1.6
```

O `styletts2` e instalado com `--no-deps` depois que as dependencias de runtime ja foram instaladas manualmente. Isso evita que o pacote rebaixe bibliotecas do Kaggle e cause conflitos.

Avisos de `pip` citando `google-cloud`, `bigquery`, `dask-cuda`, `jax`, `opencv` ou pacotes semelhantes sao do ambiente global do Kaggle. Este projeto nao usa Google Drive nem Colab.

## Geracao do audio

A celula de geracao usa:

```python
texto = 'Digite aqui o texto que voce quer transformar em audio.'
audio_path = synthesize_for_notebook(synthesizer, texto)
```

Ela mostra:

- player para ouvir o audio;
- link `Download do WAV`;
- caminho do arquivo gerado.

Os WAVs ficam em:

```text
/kaggle/working/audios_gerados
```

## Gradio

A interface Gradio e opcional. Ela fica na ultima celula. Enquanto ela estiver rodando, o notebook fica ocupado. Para gerar varios audios sem travar o fluxo, use a celula de geracao direta.
