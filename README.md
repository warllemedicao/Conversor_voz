# Conversor de Voz Neural no Colab

## Historico de Atualizacoes

### 2026-06-10 - Kaggle F5-TTS

- Adicionado fluxo Kaggle exclusivo para F5-TTS usando o repositorio `warllem/Super_voz` no Hugging Face.
- Removida do fluxo Kaggle a logica antiga baseada em checkpoints StyleTTS2.
- O carregador Kaggle agora le `voices/v_minha_voz_f5_tts_ptbr/manifest.json`, seleciona `model/model_2000.pt`, baixa apenas os artefatos necessarios e valida vocabulario, referencia, vocoder e arquitetura antes da inferencia.
- O WAV de teste esperado no Kaggle e `/kaggle/working/resultado_voz.wav`.

## Contexto original Colab

Projeto criado para executar, no Google Colab, uma voz neural treinada a partir da pasta publica do Google Drive:

`https://drive.google.com/drive/folders/13uBrrPLx--PlHho0fcz5xW8eYehimRcC?usp=sharing`

O arquivo preferencial procurado e `epoch_2nd_00024.pth`. O notebook tambem aceita o nome antigo `neuralepoch_2nd_00024.pth`.
O arquivo de configuracao YAML usado pelo notebook e baixado deste link:

`https://drive.google.com/file/d/1y_fKsgq8h_uWVCPDmzc9bnR2vmnJA1Pb/view?usp=sharing`

## Como usar

1. Abra o notebook `conversor_voz_one_click_colab.ipynb` no Google Colab.
2. Ative GPU em `Ambiente de execucao > Alterar tipo de ambiente de execucao > T4 GPU`.
3. Execute as celulas em ordem.
4. Autorize o login do Google Drive quando a celula de montagem solicitar.
5. Quando a interface abrir, digite uma frase e pressione Enter.
6. O audio gerado aparece como player e como arquivo para download.

## Arquivos do projeto

- `conversor_voz_one_click_colab.ipynb`: notebook one-click para Colab.
- `conversor_voz_colab.py`: modulo Python com download, deteccao do modelo, sintese e interface.
- `conversor_voz_requirements_colab.txt`: dependencias instaladas pelo notebook.

Cada arquivo principal possui um `.md` ao lado explicando sua funcao.

## Observacao tecnica importante

Um arquivo `.pth` sozinho normalmente nao basta para sintetizar voz por texto. O runtime tambem precisa saber qual arquitetura gerou esse checkpoint. O YAML informado contem configuracao de StyleTTS2, entao o projeto agora tenta StyleTTS2 antes dos formatos alternativos:

- StyleTTS2: `.pth` + `.yml/.yaml`.
- Coqui TTS/VITS/YourTTS: `config.json` + `.pth`.
- Piper: `.onnx` + `.json`.

Se ainda faltar algum submodelo citado pelo YAML, como `Utils/ASR/epoch_00080.pth`, `Utils/JDC/bst.t7` ou `Utils/PLBERT/`, o erro indicara que esses arquivos auxiliares precisam estar disponiveis junto do checkpoint.

Se aparecer `ValueError: numpy.dtype size changed`, significa que o runtime ficou com uma instalacao binaria inconsistente do NumPy apos instalar dependencias cientificas. O notebook reinstala `numpy==1.26.4` antes de importar `styletts2` para reduzir esse conflito.

Se aparecer `UnpicklingError` relacionado a `weights_only`, isso vem da mudanca de padrao do PyTorch 2.6 ao carregar checkpoints. O carregador StyleTTS2 do projeto aplica `weights_only=False` durante a inicializacao do modelo, assumindo que os checkpoints usados sao confiaveis.
