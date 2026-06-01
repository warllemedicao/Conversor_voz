# conversor_voz_one_click_colab.ipynb

Notebook one-click para Google Colab.

Ao executar todas as celulas, ele:

1. Instala dependencias de sistema e Python.
2. Cria o arquivo `conversor_voz_colab.py` dentro do runtime do Colab.
3. Baixa a pasta do Google Drive e o arquivo YAML de configuracao do StyleTTS2.
4. Detecta o modelo treinado, priorizando `neuralepoch_2nd_00024.pth`.
5. Abre a interface para digitar texto e gerar um arquivo WAV.

O notebook e autocontido: mesmo que apenas ele seja aberto no Colab, ele grava o modulo Python necessario antes de executar o programa.
