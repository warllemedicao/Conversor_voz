# Projeto Super Voz no Kaggle

Este documento descreve a versao Kaggle do projeto e o comportamento esperado.

## Estrutura da pasta

```text
README_kaggle.md
PROJETO_KAGGLE.md
conversor_voz_kaggle.ipynb
conversor_voz_kaggle.py
conversor_voz_requirements_kaggle.txt
```

Nao ha arquivos Colab, audios locais ou pesos versionados nessa pasta.

## Fluxo atual

O notebook foi simplificado para funcionar com `Run All`:

1. Prepara GPU/token e instala dependencias. Verifica se a GPU e compativel (avisa se for Tesla P100).
2. Cria `/kaggle/working/conversor_voz_kaggle.py`.
3. Baixa o pacote `warllem/Super_voz`:
   - Etapa 1: Baixa metadados, logs e arquivos auxiliares (leves).
   - Etapa 2: Analisa o melhor checkpoint e baixa **apenas** o arquivo `.pth` selecionado (evitando download de ~9GB).
4. Carrega a voz e gera um audio.
5. Permite gerar outro audio sem recarregar.
6. Opcionalmente abre Gradio.

Se qualquer etapa falhar, o notebook para e salva o traceback em:

```text
/kaggle/working/super_voz_kaggle.log
```

## Por que o fluxo mudou

As versoes anteriores tentavam reinstalar `numpy`, `scipy` e `pandas` dentro do notebook. No Kaggle, isso e instavel porque o kernel e varias bibliotecas pre-instaladas ja podem ter carregado NumPy em memoria. Trocar essas bibliotecas durante a sessao causou erros como:

```text
numpy.dtype size changed
cannot load module more than once per process
AttributeError: module 'numpy' has no attribute '_no_nep50_warning'
```

Por isso, a versao atual detecta as versoes nativas do Kaggle (ex: `numpy==1.26.4`), as salva em um arquivo de `constraints.txt` e instala as outras dependencias sem o parametro `-U` (Upgrade). Isso garante que o ambiente seja preparado rapidamente usando o cache do Kaggle e que as bibliotecas base nunca sejam alteradas. Ela instala `styletts2==0.1.6` com `--no-deps`.

**Importante sobre GPU**: A GPU **Tesla P100** (sm_60) nao e compativel com as versoes recentes do PyTorch/StyleTTS2 (exige sm_70+). O notebook agora avisa para trocar para a **Tesla T4**.

## Origem dos arquivos

O modelo vem do Hugging Face:

```text
https://huggingface.co/warllem/Super_voz
```

Repositorio:

```text
warllem/Super_voz
```

Destino no Kaggle:

```text
/kaggle/working/Super_voz
```

Download seletivo (Otimizado):
- Metadados e Utils primeiro.
- Checkpoint selecionado individualmente (ex: `best_model.pth`).


## Arquivos corretos

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

O checkpoint principal usado e:

```text
model/best_model.pth
```

## Audio de referencia

O audio esperado e:

```text
data_reference/referencia_voz.wav
```

O codigo tambem procura logs com `.wav` e metricas. Se encontrar, pode escolher o melhor audio pelo score. Se nao encontrar, usa `referencia_voz.wav`.

## Saida

Audios gerados:

```text
/kaggle/working/audios_gerados
```

A celula de geracao mostra player e link `Download do WAV`.
