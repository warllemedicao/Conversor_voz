# Voz_Noslen F5-TTS ONNX (Modo Turbo)

Versão atual do packager: `2026.06.19.turbo.v6`.

Este diretório contém as ferramentas para gerar o pacote **Turbo** do modelo F5-TTS `Voz_Noslen`. O pacote é projetado para execução eficiente em CPU (ONNX Runtime) e deploy em ambientes serverless como o Google Cloud Run.

## Arquitetura Turbo
Diferente do fluxo F5-TTS completo, a arquitetura Turbo exporta somente o núcleo do Transformer (**Diffusion Transformer - DiT**) como um grafo ONNX estático. Ele **não** é um export text-to-audio end-to-end.

- **Fidelidade:** Mantém a precisão original (FP32).
- **Flexibilidade:** Preprocessamento, tokenização, loop de difusão/ODE, Vocos e escrita do WAV permanecem no backend Python.
- **Portabilidade:** Gera um artefato minimalista com metadados integrados.
- **Qualidade:** Não aplica quantização, pruning, troca de vocoder ou redução agressiva de steps.

## Fluxo Kaggle Ponta a Ponta
1. O notebook baixa a árvore `voices/v_minha_voz_f5_tts_ptbr` do Hugging Face Storage Bucket `https://huggingface.co/buckets/warllem/Voz_Noslen`.
2. O packager procura manifesto fonte quando existir (`manifest.json` ou `metadata.json`), mas não depende dele.
3. A seleção de checkpoint é determinística: manifesto, depois `model_last.pt`, depois o maior `model_*.pt`.
4. `vocab.txt` e `data_reference/referencia_voz.wav` são validados antes da exportação.
5. O F5-TTS é carregado em CPU com `mel_spec_type="vocos"` e arquitetura F5-TTS v1 Base.
6. O wrapper ONNX exporta apenas o passo do DiT core.
7. O smoke test ONNX abre o grafo no `onnxruntime` CPU, valida inputs/outputs, executa um `sess.run` dummy e registra tempo do core.
8. Um smoke test separado gera `validation/full_pipeline_smoke.wav` pelo pipeline Python F5-TTS + Vocos, para provar que a geração completa de WAV continua funcionando.
9. Só depois de `validation.json -> status: verified` o notebook compacta e envia o pacote.

## Estrutura do Pacote Gerado
O packager cria uma pasta isolada `onnx_package_turbo_<timestamp>/` com:

```text
├── onnx/
│   └── f5_tts_transformer_core.onnx  <-- Grafo DiT core, nao end-to-end
├── model/
│   └── vocab.txt                    <-- Dicionário de tokens
├── reference/
│   └── referencia_voz.wav           <-- Áudio base para clonagem
├── manifest.json                    <-- Versão, backend e lista de arquivos
├── metadata.json                    <-- Contrato ONNX (inputs/outputs) e shapes
└── validation.json                  <-- Relatorio com smoke tests ONNX core e WAV Python
```

## Contrato ONNX (Interface)
O arquivo `f5_tts_transformer_core.onnx` segue estritamente este contrato:

| Entrada | Tipo | Shape | Descrição |
| :--- | :--- | :--- | :--- |
| `x` | float32 | `[1, 128, 100]` | Tensor latente (ruído) |
| `cond` | float32 | `[1, 128, 100]` | Condicionamento Mel |
| `text_ids` | int64 | `[1, text_len]` | IDs do texto alvo |
| `text_lengths` | int64 | `[1]` | Comprimento real do texto |
| `time_steps` | float32 | `[1]` | Passo de tempo da difusão |

**Saída:**
- `dx` (float32): Velocidade prevista para o próximo passo, shape `[1, 128, 100]`.

Observação: na exportação ONNX via tracer legado, o F5-TTS fixa internamente o comprimento de áudio em `128` frames por causa de `seq_len.max().item()` no `TextEmbedding`. O backend deve processar/chunkar chamadas Turbo nesse tamanho.

O contrato acima é um contrato de **núcleo de modelo**. O Lite CPU deve preparar `x`, `cond`, `text_ids`, `text_lengths` e `time_steps`, chamar o ONNX para obter `dx`, e manter em Python o restante do pipeline F5-TTS/Vocos.

## Como Gerar
1. Abra `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` no Kaggle.
2. Certifique-se de que a **Internet** está ligada.
3. Habilite o secret `HF_TOKEN` em `Add-ons -> Secrets`.
4. Execute todas as células.
5. O notebook valida o ONNX, gera o `.zip` e faz upload direto para o Hugging Face.

Destino padrão do upload:

```text
repo_id: warllem/Voz_Noslen_Turbo
repo_type: model
pasta: turbo/
privado: sim
```

Esses valores podem ser alterados no Kaggle com variáveis de ambiente:

```text
HF_UPLOAD_REPO_ID
HF_UPLOAD_REPO_TYPE
HF_UPLOAD_FOLDER
HF_PRIVATE_REPO
```

## Regras de Engenharia
- **Isolamento:** Nunca altera os arquivos originais em `voices/`.
- **Sincronia:** O notebook gera automaticamente o script `.py` para garantir que a lógica de exportação esteja sempre atualizada.
- **Validação:** O pacote só é considerado "Pronto" se passar no teste ONNX Runtime CPU e no smoke test de WAV do pipeline Python.
- **Contrato DiT:** O wrapper Turbo chama o Transformer com argumentos nomeados (`x`, `cond`, `text`, `time`). `text_lengths` permanece como entrada ONNX ancorada no grafo, mas não é passado como quinto argumento posicional para evitar que o F5-TTS o interprete como `drop_audio_cond`/máscara de áudio.
- **Máscara de áudio:** O wrapper cria uma máscara 2D `[batch, duration]` a partir de `x` e a passa como `mask` quando a versão instalada do F5-TTS suporta esse argumento.
- **Duração fixa:** A versão ONNX v6 usa `TURBO_DURATION=128`; não marcar `x`, `cond` ou `dx` como dinâmicos no ONNX.

## Arquivos Canônicos
Os arquivos canônicos atuais são:

- `kaggle/f5_tts_onnx_packager_kaggle.py`
- `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb`

Também existem aliases de compatibilidade:

- `kaggle/conversor_voz_kaggle.py`
- `kaggle/conversor_voz_kaggle.ipynb`
