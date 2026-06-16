import json
from pathlib import Path

# Carrega o conteúdo do script
script_path = Path('kaggle/f5_tts_onnx_packager_kaggle.py')
script_content = script_path.read_text(encoding='utf-8')

# Formata o conteúdo do script como uma lista de strings JSON-safe
# Isso evita problemas com aspas triplas e caracteres especiais
script_json_lines = json.dumps(script_content.splitlines(keepends=True))

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
        'source': ['!pip install -q f5-tts>=1.1.9 vocos>=0.1.0 onnx>=1.16.0 onnxruntime>=1.18.0 onnxconverter-common requests huggingface_hub']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import json\n',
            'from pathlib import Path\n',
            '\n',
            '# Criamos o script auxiliar no worker do Kaggle de forma segura\n',
            'script_lines = ' + script_json_lines + '\n',
            '\n',
            'Path(\"/kaggle/working/f5_tts_onnx_packager_kaggle.py\").write_text(\"\".join(script_lines), encoding=\"utf-8\")\n',
            'print(\"Script f5_tts_onnx_packager_kaggle.py criado com sucesso.\")'
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
