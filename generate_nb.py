import json
from pathlib import Path

# Versão do empacotador para controle no Kaggle
VERSION = "2026.06.16.4"

# Carrega o conteúdo do script
script_path = Path('kaggle/f5_tts_onnx_packager_kaggle.py')
script_content = script_path.read_text(encoding='utf-8')

# Divide o script em linhas para embutir no notebook de forma segura
script_lines = script_content.splitlines(keepends=True)

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# Voz_Noslen F5-TTS ONNX \"Modo Turbo\" - v' + VERSION + '\n',
            '\n',
            'Este notebook foi atualizado para corrigir erros de dependência e indentação.\n',
            '\n',
            '**Instruções:**\n',
            '1. Vá em **Settings** -> **Internet** e mude para **On**.\n',
            '2. Se o erro `onnxruntime-quantization` persistir, você está rodando uma versão antiga do notebook.\n',
            '3. Execute as células em ordem.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 1) Instalação de Dependências Otimizada\n',
            '# Nota: onnxruntime já inclui ferramentas de quantização. onnxruntime-quantization NÃO deve ser instalado separadamente.\n',
            '!pip install -q f5-tts>=1.1.9 vocos>=0.1.0 onnx>=1.16.0 onnxruntime>=1.18.0 onnxconverter-common requests huggingface_hub'
        ]
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
            '# 2) Criação do Script de Processamento\n',
            '# O script é embutido como uma lista de linhas para evitar estouro de buffer no editor do Kaggle\n',
            'script_data = ' + json.dumps(script_lines, indent=4) + '\n',
            '\n',
            'output_path = Path(\"/kaggle/working/f5_tts_onnx_packager_kaggle.py\")\n',
            'output_path.write_text(\"\".join(script_data), encoding=\"utf-8\")\n',
            'print(f\"Script criado com sucesso em: {output_path} (Versão: ' + VERSION + ')\")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 3) Configurações de Origem e Destino\n',
            'import os\n',
            'from datetime import datetime, timezone\n',
            'from kaggle_secrets import UserSecretsClient\n',
            '\n',
            'try:\n',
            '    os.environ[\"HF_TOKEN\"] = UserSecretsClient().get_secret(\"HF_TOKEN\")\n',
            'except: \n',
            '    print(\"AVISO: HF_TOKEN não encontrado no Kaggle Secrets. O upload falhará.\")\n',
            '\n',
            'os.environ[\"HF_SOURCE_URL\"] = \"https://huggingface.co/buckets/warllem/Voz_Noslen\"\n',
            'os.environ[\"HF_VOICE_DIR\"] = \"voices/v_minha_voz_f5_tts_ptbr\"\n',
            'os.environ[\"HF_UPLOAD_REPO_ID\"] = \"warllem/Voz_Noslen_ONNX\"\n',
            'os.environ[\"HF_TARGET_FOLDER\"] = \"onnx_packages/turbo_\" + datetime.now(timezone.utc).strftime(\"%Y%m%d_%H%M%S\")\n',
            'print(\"Configurações carregadas.\")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 4) Execução do Empacotador Turbo\n',
            'import subprocess, sys\n',
            'print(\"Iniciando exportação End-to-End + Quantização INT8...\")\n',
            'process = subprocess.Popen([sys.executable, \"f5_tts_onnx_packager_kaggle.py\"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n',
            'for line in process.stdout: \n',
            '    print(line, end=\"\")'
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

print(f"Notebook gerado com sucesso (Versão {VERSION}).")
