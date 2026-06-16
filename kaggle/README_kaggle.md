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

## Empacotar Voz_Noslen em ONNX 'Modo Turbo' (v2026.06.16.FINAL)

Use o notebook `voz_noslen_f5_tts_onnx_kaggle.ipynb` para criar um pacote otimizado da voz F5-TTS em `warllem/Voz_Noslen_ONNX`.

O **Modo Turbo** traz as seguintes melhorias:
1. **Wrapper End-to-End**: O arquivo ONNX agora encapsula o Transformer (DiT), o ODE Solver (Euler) e o Vocoder (Vocos). O contrato de entrada aceita `text` (IDs), `x` (noise), `cond` (mel), `time_steps` e `mask`, devolvendo o áudio pronto.
2. **Quantização INT8**: O modelo é reduzido de ~2.3GB (FP32) para **~1.2GB (INT8)**, permitindo rodar em ambientes com pouca RAM (como Cloud Run) e acelerando a inferência em CPU.
3. **Gestão de Memória**: O processo de exportação no Kaggle foi otimizado para rodar em CPU e liberar RAM imediatamente após a geração do ONNX, permitindo que a quantização ocorra sem estourar os limites da plataforma.

### Como rodar no Kaggle:
1.  Acesse o arquivo `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` no GitHub.
2.  Clique em **"Raw"** e **copie todo o texto**.
3.  No Kaggle, **cole** o conteúdo em um novo notebook.
4.  Ative a **Internet** (painel lateral).
5.  Clique em **Run All**.

O processo é totalmente automatizado e gera o pacote completo pronto para deploy.
