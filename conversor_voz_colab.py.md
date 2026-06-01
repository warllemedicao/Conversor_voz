# conversor_voz_colab.py

Modulo principal do programa.

Ele faz cinco tarefas:

1. Importa os arquivos do modelo para `/content/voz_neural`.
   Primeiro procura o checkpoint no Drive montado e copia a pasta encontrada. Se nao encontrar, baixa a pasta publica do Google Drive usando `gdown`.
2. Procura automaticamente o modelo de voz, priorizando `epoch_2nd_00024.pth` e tambem aceitando `neuralepoch_2nd_00024.pth`.
3. Baixa tambem o arquivo YAML informado separadamente:
   `https://drive.google.com/file/d/1y_fKsgq8h_uWVCPDmzc9bnR2vmnJA1Pb/view?usp=sharing`
4. Carrega o modelo se houver um formato suportado:
   - StyleTTS2: `.pth` com `.yml/.yaml`.
   - Coqui TTS: `.pth` com `config.json`.
   - Piper: `.onnx`.
5. Abre uma interface Gradio onde o usuario digita uma frase, pressiona Enter e recebe um arquivo `.wav` para ouvir e baixar.

O YAML lido indica StyleTTS2 porque contem chaves como `ASR_config`, `PLBERT_dir`, `model_params` e `preprocess_params`.
