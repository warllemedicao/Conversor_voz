# Super Voz F5-TTS no Kaggle

Use `conversor_voz_kaggle.ipynb` em um notebook Kaggle com GPU e internet ligadas.

## Historico

### 2026-06-10

- Fluxo Kaggle migrado para F5-TTS exclusivo.
- Download antigo por padroes `model/**` foi substituido por auditoria do manifesto em `voices/v_minha_voz_f5_tts_ptbr/manifest.json`.
- Removido fallback silencioso para caminhos de estruturas antigas.
- Adicionada auditoria leve antes da inferencia e validacao de vocabulario, referencia de audio, arquitetura, vocoder e checkpoint.
- Notebook atualizado para gerar `/kaggle/working/resultado_voz.wav` com o texto de teste `Boa noite Warllem, sua voz esta pronta.`

## Como rodar

1. Ative GPU, preferencialmente Tesla T4 ou superior.
2. Se o Hugging Face exigir autenticacao, adicione um Kaggle Secret chamado `HF_TOKEN`.
3. Execute as celulas em ordem.
4. Rode primeiro a auditoria leve. Ela baixa apenas manifesto, vocabulos, configuracao pequena e referencia.
5. Rode a inferencia. Ela baixa somente o checkpoint escolhido pelo manifesto e gera `/kaggle/working/resultado_voz.wav`.

O log completo fica em:

```text
/kaggle/working/super_voz_kaggle.log
```

## Estrutura remota atual

Repositorio:

```text
warllem/Super_voz
```

Pacote de voz usado:

```text
voices/v_minha_voz_f5_tts_ptbr/
```

Arquivos de inferencia:

```text
voices/v_minha_voz_f5_tts_ptbr/manifest.json
voices/v_minha_voz_f5_tts_ptbr/model/model_2000.pt
voices/v_minha_voz_f5_tts_ptbr/model/latest_checkpoint.pt
voices/v_minha_voz_f5_tts_ptbr/model/model_last.pt
voices/v_minha_voz_f5_tts_ptbr/model/vocab.txt
voices/v_minha_voz_f5_tts_ptbr/data_reference/referencia_voz.wav
libraries/f5_tts_ptbr_tharyck/setting.json
libraries/f5_tts_ptbr_tharyck/vocab.txt
libraries/f5_tts_ptbr_tharyck/model_last.safetensors
```

O manifesto atual seleciona `model/model_2000.pt`. Esse arquivo e `latest_checkpoint.pt` apontam para o mesmo objeto LFS no Hugging Face. O carregador trabalha apenas com F5-TTS e nao usa a estrutura antiga.

## Politica de download

Por padrao, o programa baixa apenas:

- `manifest.json`;
- `setting.json`;
- `vocab.txt` da voz;
- `vocab.txt` da biblioteca-base para comparacao;
- audio de referencia;
- checkpoint selecionado pelo manifesto.

Ele nao baixa todos os checkpoints grandes para diagnostico.

## Diagnostico

A auditoria imprime uma tabela com:

```text
Manifesto
Checkpoint principal
Arquitetura identificada
Vocab encontrado
Vocab compativel
Referencia de audio
Texto da referencia
Vocoder
CUDA
Checkpoint legivel
Inferencia pronta
```

O pacote remoto nao contem um `.txt` ao lado de `referencia_voz.wav`. Nesse caso, a inferencia deixa o F5-TTS transcrever a referencia automaticamente com ASR.

## Empacotar Voz_Noslen em ONNX/Lite

Use `f5_tts_onnx_packager_kaggle.py` para criar um pacote versionado da voz F5-TTS em `warllem/Voz_Noslen_ONNX`.

O script nao altera os arquivos remotos existentes da voz original. Ele baixa somente os arquivos necessarios, copia a origem para `/kaggle/working/voz_noslen_onnx_package/f5_tts_original`, cria caminhos simples para runtime em `model/` e `reference/`, exporta um ONNX do nucleo Transformer/DiT para `onnx/`, cria `manifest.json`, `onnx_export_report.json`, `package_metadata.json` e adiciona `scripts/test_package_cpu.py`.

O caminho recomendado e rodar o notebook:

```text
kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb
```

Esse notebook:

- embute a versao atual do empacotador no worker `/kaggle/working`;
- usa caches gravaveis para `numba` e `matplotlib`;
- evita reinstalar `torch/torchaudio` se o Kaggle ja tiver PyTorch;
- inclui uma celula opcional de keep-alive/heartbeat do navegador;
- executa o empacotamento com logs periodicos durante celulas longas.

Na celula 1, confirme que aparece:

```text
Packager version esperada: 2026.06.16.1
```

Se aparecer `2026.06.15.6`, o Kaggle ainda esta usando uma copia antiga do notebook/script.

A pasta de destino padrao e:

```text
onnx_packages/voz_noslen_f5tts_onnx_<data_hora>
```

## Estrutura esperada do pacote ONNX/Lite

Um pacote validado com a versao `2026.06.16.1` deve conter, no minimo:

```text
manifest.json
package_metadata.json
onnx_export_report.json
onnx/f5_tts_transformer_core.onnx
model/model_2000.pt
model/vocab.txt
reference/referencia_voz.wav
reference/reference_text.txt
scripts/test_package_cpu.py
test_outputs/voz_noslen_lite_cpu.wav
f5_tts_original/...
```

O `onnx_export_report.json` deve registrar:

- `packager_version: "2026.06.16.1"`;
- inputs/outputs do ONNX com nomes, shapes e tipos;
- `pipeline_contract.full_text_to_audio_onnx_available: false`;
- resultado de `onnxruntime_cpu_smoke_test`;
- resultado de `wav_generation_cpu_test`;
- lista `generated_files`;
- comando exato de teste em `cpu_test_command`.

Pacotes publicados antes do commit `7bbadc4`, por exemplo `onnx_packages/voz_noslen_f5tts_onnx_20260616_020835`, foram gerados com `packager_version: "2026.06.15.6"`. Eles contem apenas `f5_tts_original/`, `onnx/f5_tts_transformer_core.onnx`, `onnx_export_report.json` e `package_metadata.json`; portanto nao sao o pacote ONNX/Lite final validado.

No Kaggle:

```bash
pip install -r /kaggle/input/seu-projeto/conversor_voz_requirements_kaggle.txt
python /kaggle/input/seu-projeto/f5_tts_onnx_packager_kaggle.py \
  --source https://huggingface.co/buckets/warllem/Voz_Noslen \
  --voice-dir voices/v_minha_voz_f5_tts_ptbr \
  --download-mode essential \
  --upload-repo-id warllem/Voz_Noslen_ONNX
```

Se o repo exigir permissao de escrita, crie um Kaggle Secret chamado `HF_TOKEN`.

Para testar sem enviar ao Hugging Face:

```bash
python /kaggle/input/seu-projeto/f5_tts_onnx_packager_kaggle.py \
  --source https://huggingface.co/buckets/warllem/Voz_Noslen \
  --voice-dir voices/v_minha_voz_f5_tts_ptbr \
  --download-mode essential \
  --no-upload
```

O teste CPU roda por padrao antes do upload. Ele valida o ONNX com `onnxruntime` e gera um WAV com a frase:

```text
Boa noite Warllem, este é um teste do modo lite em CPU.
```

Comando gerado dentro do pacote:

```bash
python scripts/test_package_cpu.py \
  --text "Boa noite Warllem, este é um teste do modo lite em CPU." \
  --output-wav test_outputs/voz_noslen_lite_cpu.wav \
  --nfe-step 4 \
  --speed 1.0
```

Use `--skip-cpu-test` somente para diagnostico. Para publicacao final, deixe o teste passar; caso contrario o pacote nao deve ser considerado validado.

O modo `essential` evita estourar o disco do Kaggle: ele ignora `.tmp`, baixa apenas a voz escolhida, preserva o `manifest.json`, `vocab.txt`, audio de referencia, docs/configs pequenas e um checkpoint principal. Use `--download-mode all` somente em um ambiente com disco suficiente.

Para preservar a qualidade, nao quantize o ONNX, mantenha FP32, use o mesmo checkpoint, o mesmo vocabulario, `F5TTS_v1_Base`, vocoder `vocos`, sample rate de 24000 Hz e a referencia de audio/texto da voz treinada.

Se a instalacao no Kaggle mostrar conflitos com `dask-cuda`, `cuml` ou `cudf`, trate como aviso do ambiente base. O erro que bloqueia a exportacao ONNX e falta de pacote como `onnxscript`; por isso ele esta listado no requirements.

O exportador tenta primeiro o caminho legado `torch.onnx.utils.export`, depois `torch.onnx.export(..., dynamo=False)` e, por ultimo, `torch.jit.trace` seguido de exportacao ONNX. As entradas flutuantes do wrapper sao convertidas para o dtype real dos pesos do modelo para evitar erro `mat1 and mat2 must have the same dtype, but got Float and Half` quando o checkpoint carregar em FP16.

Limitacao importante: o ONNX exportado continua sendo apenas o nucleo DiT/Transformer, com entradas de baixo nivel como `x`, `cond`, `text`, `time` e `mask`. O F5-TTS completo exige tokenizacao/preprocessamento, condicionamento por audio de referencia, loop iterativo de flow matching, vocoder e escrita WAV. Por isso o pacote documenta um pipeline parcial: `onnxruntime` valida o nucleo ONNX, mas a geracao texto->WAV usa runtime Python `f5-tts` + `vocos` em CPU.
