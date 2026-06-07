# Super Voz no Kaggle

Use `conversor_voz_kaggle.ipynb` em um notebook Kaggle com GPU ligada.

## Secret do Hugging Face

Crie um secret no Kaggle com um destes nomes:

- `HF_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`

O notebook usa esse token para baixar `warllem/Super_voz` via `huggingface_hub.snapshot_download`. Os audios e pesos nao ficam salvos neste repositorio GitHub; eles sao baixados dentro do Kaggle durante a execucao.

O download seletivo traz os arquivos necessarios para inferencia e analise:

```text
model/**
docs/**
inference/**
tokenizer/**
data_reference/referencia_voz.wav
data_reference/*.txt
data_reference/*.csv
```

## Arquivos detectados no Hugging Face

O pacote principal de inferencia fica em:

- `model/config.yml`
- `model/best_metric.txt`
- `model/best_model.pth`
- `model/latest_checkpoint.pth`
- `model/latest_checkpoint.txt`
- `model/Utils/ASR/epoch_00080.pth`
- `model/Utils/JDC/bst.t7`
- `model/Utils/PLBERT/step_1000000.t7`
- `data_reference/referencia_voz.wav`

`best_metric.txt` aponta que o melhor treinamento veio de `epoch_2nd_00045.pth`, com `validation_loss=0.268`. No pacote final, esse checkpoint esta salvo como `model/best_model.pth`.

## Selecao automatica

O modulo `conversor_voz_kaggle.py` primeiro le `best_metric.txt`; se ele existir, usa `model/best_model.pth`. Se esse arquivo faltar, ele le os `train.log` e seleciona o epoch com menor `Validation loss`. O audio de referencia tambem e escolhido por logs quando houver uma linha com WAV e metrica; se nao houver, usa `data_reference/referencia_voz.wav`.

## Geracao e download do audio

A forma recomendada no Kaggle e usar a celula simples do notebook:

```python
texto = 'Digite aqui o texto que voce quer transformar em audio.'
audio_path = synthesize_for_notebook(synthesizer, texto)
```

Essa celula gera o WAV, mostra um player e cria um link `Download do WAV` no proprio output do notebook. Os arquivos ficam em:

```text
/kaggle/working/audios_gerados
```

A interface Gradio continua disponivel como opcional, mas enquanto ela estiver rodando a celula fica presa. Para gerar varios audios sem parar o notebook, use a celula `synthesize_for_notebook`.

Observacao: o pacote `styletts2` deve ser instalado como `styletts2==0.1.6`, porque essa e a versao disponivel no PyPI usado pelo Kaggle.

Na primeira vez que a celula 5 instala `numpy==1.26.4`, `scipy==1.12.0` e `pandas==2.2.2`, ela reinicia o kernel automaticamente. Isso e necessario porque NumPy/SciPy sao extensoes nativas e nao podem ser recarregadas com seguranca no mesmo processo Python. Depois que o Kaggle reconectar, clique em `Run All` novamente ou execute desde a celula 1; a celula 5 vai detectar o marcador em `/kaggle/working/.super_voz_deps_v4_installed` e pular a reinstalacao.

A celula 5 e autocontida: se o Kaggle tentar continuar nela depois do reinicio e a variavel `bundle` nao existir, ela importa o modulo, confere o download em `/kaggle/working/Super_voz` e detecta o modelo novamente.

Se aparecer `numpy.dtype size changed`, erro vindo de `scipy`, ou `cannot load module more than once per process`, reinicie o kernel/runtime do Kaggle e execute tudo em ordem.

Mensagens de `pip` citando `google-cloud`, `bigquery`, `dask-cuda`, `jax`, `opencv` ou pacotes parecidos sao avisos do ambiente global do Kaggle, que ja vem com muitas bibliotecas instaladas. Isso nao significa que o projeto esta usando Google Drive ou Colab. O notebook usa Hugging Face para baixar o modelo.
