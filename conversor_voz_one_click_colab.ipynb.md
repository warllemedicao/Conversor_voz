# conversor_voz_one_click_colab.ipynb

Notebook one-click para Google Colab.

Este e o unico notebook usado pelo projeto. Ao executar as celulas em ordem, ele:

1. Instala dependencias de sistema e Python.
2. Faz login e monta o Google Drive em `/content/drive`.
3. Cria o arquivo `conversor_voz_colab.py` dentro do runtime do Colab.
4. Importa os arquivos do modelo para `/content/voz_neural`.
   - Primeiro procura `neuralepoch_2nd_00024.pth` dentro do Drive montado.
   - Se nao encontrar no Drive montado, baixa a pasta publica com `gdown`.
   - Baixa o YAML de configuracao e salva junto dos arquivos do modelo.
5. Carrega a voz detectada.
6. Abre a interface para digitar texto e gerar um arquivo WAV.

O notebook e autocontido: mesmo que apenas ele seja aberto no Colab, ele grava o modulo Python necessario antes de executar o programa.
