"""Compatibilidade para o nome antigo do fluxo Kaggle.

O packager canonico e `f5_tts_onnx_packager_kaggle.py`. Este arquivo existe
para comandos antigos que ainda chamam `conversor_voz_kaggle.py`.
"""

from f5_tts_onnx_packager_kaggle import main


if __name__ == "__main__":
    main()
