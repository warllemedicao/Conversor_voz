import json
from pathlib import Path

script_content = Path('kaggle/f5_tts_onnx_packager_kaggle.py').read_text(encoding='utf-8')

# Ensure script_content is escaped correctly for insertion into a JSON-based notebook
# We will use the list of strings format for the cell source to avoid most escaping issues.
script_lines = [line + '\n' for line in script_content.split('\n')]

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# Voz_Noslen F5-TTS ONNX \"Modo Turbo\"\n',
            '\n',
            'Este notebook implementa a exportação End-to-End do F5-TTS para ONNX com quantização INT8. \n',
            'O objetivo é reduzir o modelo de ~5.39GB para ~1.2GB e permitir inferência rápida em CPU no Cloud Run.\n',
            '\n',
            '**Requisitos:**\n',
            '- Ative Internet no Kaggle.\n',
            '- Use GPU para acelerar o carregamento inicial.\n',
            '- Secret `HF_TOKEN` configurado.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': ['!pip install -q f5-tts>=1.1.9 vocos>=0.1.0 onnx>=1.16.0 onnxruntime>=1.18.0 onnxconverter-common']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'from pathlib import Path\n',
            '# Criamos o script auxiliar no worker do Kaggle\n',
            'script_content = r\"\"\"' + script_content + '\"\"\"\n',
            'Path(\"/kaggle/working/f5_tts_onnx_packager_kaggle.py\").write_text(script_content, encoding=\"utf-8\")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import os\n',
            'from datetime import datetime, timezone\n',
            'from kaggle_secrets import UserSecretsClient\n',
            'try:\n',
            '    os.environ[\"HF_TOKEN\"] = UserSecretsClient().get_secret(\"HF_TOKEN\")\n',
            'except: pass\n',
            'os.environ[\"HF_SOURCE_URL\"] = \"https://huggingface.co/buckets/warllem/Voz_Noslen\"\n',
            'os.environ[\"HF_VOICE_DIR\"] = \"voices/v_minha_voz_f5_tts_ptbr\"\n',
            'os.environ[\"HF_UPLOAD_REPO_ID\"] = \"warllem/Voz_Noslen_ONNX\"\n',
            'os.environ[\"HF_TARGET_FOLDER\"] = \"onnx_packages/turbo_\" + datetime.now(timezone.utc).strftime(\"%Y%m%d_%H%M%S\")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import subprocess, sys\n',
            'process = subprocess.Popen([sys.executable, \"f5_tts_onnx_packager_kaggle.py\"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n',
            'for line in process.stdout: print(line, end=\"\")'
        ]
    }
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
