# conversor_voz_requirements_colab.txt

Lista as bibliotecas Python usadas pelo notebook no Google Colab.

No notebook atual, as dependencias sao instaladas depois que os arquivos do modelo ja foram importados para `/content/voz_neural`.

- `gdown`: baixa a pasta publica do Google Drive.
- `gradio`: cria a caixa de texto e os componentes de audio/download.
- `pydub` e `soundfile`: manipulam arquivos de audio.
- `pyyaml`: dependencia comum para ler configuracoes YAML.
- `numpy==1.26.4`: versao fixada para evitar incompatibilidade binaria `numpy.dtype size changed` no Colab.
- `styletts2`: carrega o checkpoint `.pth` quando ele vem de um treinamento StyleTTS2 com `.yml/.yaml`.
- `coqui-tts`: tenta carregar modelos TTS salvos como `.pth` com `config.json`.
- `piper-tts`: suporte opcional para modelos Piper em `.onnx`.
