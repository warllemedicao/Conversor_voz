# conversor_voz_one_click_colab.ipynb

Notebook one-click para Google Colab.

Este e o unico notebook usado pelo projeto. Ao executar as celulas em ordem, ele:

1. Faz login e monta o Google Drive em `/content/drive`.
2. Cria o arquivo `conversor_voz_colab.py` dentro do runtime do Colab.
3. Importa os arquivos do modelo para `/content/voz_neural`.
   - Primeiro procura `epoch_2nd_00024.pth` ou `neuralepoch_2nd_00024.pth` dentro do Drive montado.
   - Se nao encontrar no Drive montado, baixa a pasta publica com `gdown`.
   - Baixa o YAML de configuracao e salva junto dos arquivos do modelo.
4. Detecta o motor do modelo.
5. Instala as dependencias basicas e somente o pacote necessario para o motor detectado.
6. Carrega a voz detectada.
7. Abre a interface para digitar texto e gerar um arquivo WAV.

O notebook e autocontido: mesmo que apenas ele seja aberto no Colab, ele grava o modulo Python necessario antes de executar o programa.

O pacote `styletts2` nao e instalado antes da importacao dos arquivos. Isso evita que uma falha de compatibilidade do pacote interrompa o notebook antes de copiar o modelo para dentro do Colab.

Para evitar o erro `ValueError: numpy.dtype size changed`, a celula de instalacao reinstala `numpy==1.26.4` depois de instalar `styletts2` e testa o import do pacote antes de carregar a voz.
