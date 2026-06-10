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
