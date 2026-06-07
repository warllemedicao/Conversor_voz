# Super Voz no Kaggle

Use `conversor_voz_kaggle.ipynb` em um notebook Kaggle com GPU e internet ligadas.

## Como rodar

1. Configure o secret `HF_TOKEN` no Kaggle com seu token do Hugging Face.
2. Abra `conversor_voz_kaggle.ipynb`.
3. Clique em `Run All`.
4. O notebook deve preparar dependencias, baixar o modelo, carregar a voz e gerar um WAV.

Se alguma etapa falhar, a execucao para e grava o traceback em:

```text
/kaggle/working/super_voz_kaggle.log
```

## Decisao importante sobre NumPy/SciPy

O notebook nao reinstala `numpy`, `scipy` nem `pandas`. O Kaggle ja carrega esse stack no ambiente do kernel, e trocar essas bibliotecas durante a sessao causa erros como:

```text
numpy.dtype size changed
cannot load module more than once per process
AttributeError: module 'numpy' has no attribute '_no_nep50_warning'
```

O erro `_no_nep50_warning` ocorre especificamente quando o `pip` atualiza o NumPy para a versao 2.0 em disco mas a versao antiga continua em memoria, ou quando ha incompatibilidade entre o NumPy novo e o SciPy antigo.

Por isso, o fluxo atual:
1. Detecta as versoes exatas de NumPy/SciPy/Pandas ja carregadas pelo Kaggle.
2. Cria um arquivo `constraints.txt` com essas versoes.
3. Instala as dependencias faltantes usando o Kaggle como cache (sem `-U`), garantindo que o NumPy/SciPy nao sejam alterados via `-c constraints.txt`.
4. Instala `styletts2==0.1.6` com `--no-deps`.

Isso torna a preparacao do ambiente muito mais rapida, evitando o download e a reinstalacao desnecessaria de centenas de megabytes.

## Origem dos arquivos

Os pesos, logs e audio de referencia vem do Hugging Face:

```text
warllem/Super_voz
```

O download seletivo traz:

```text
model/**
docs/**
inference/**
tokenizer/**
data_reference/referencia_voz.wav
data_reference/*.txt
data_reference/*.csv
```

O download fica em:

```text
/kaggle/working/Super_voz
```

## Arquivos esperados

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

O melhor checkpoint vem de `model/best_metric.txt` e e usado como:

```text
model/best_model.pth
```

## Geracao do audio

A celula principal gera um WAV usando o texto:

```python
texto = 'Digite aqui o texto que voce quer transformar em audio.'
```

Ela mostra player e link `Download do WAV`. Os arquivos ficam em:

```text
/kaggle/working/audios_gerados
```

Para gerar outro audio, altere o texto na celula seguinte e rode apenas essa celula, sem recarregar a voz.

## Gradio

A ultima celula abre Gradio opcionalmente. Enquanto ela estiver rodando, o notebook fica ocupado.
