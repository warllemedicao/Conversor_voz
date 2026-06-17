# Voz_Noslen F5-TTS ONNX (Modo Turbo)

Este diretório contém as ferramentas para gerar o pacote **Turbo** do modelo F5-TTS `Voz_Noslen`. O pacote é projetado para execução eficiente em CPU (ONNX Runtime) e deploy em ambientes serverless como o Google Cloud Run.

## Arquitetura Turbo
Diferente do fluxo F5-TTS completo, a arquitetura Turbo exporta o núcleo do Transformer (**Diffusion Transformer - DiT**) como um grafo ONNX estático.

- **Fidelidade:** Mantém a precisão original (FP32).
- **Flexibilidade:** O loop de inferência (Solver ODE) e o Vocoder permanecem no backend (Python), permitindo ajustes finos de qualidade e performance sem necessidade de re-exportar o modelo.
- **Portabilidade:** Gera um artefato minimalista com metadados integrados.

## Estrutura do Pacote Gerado
O packager cria uma pasta isolada `onnx_package_turbo_<timestamp>/` com:

```text
├── onnx/
│   └── f5_tts_transformer_core.onnx  <-- Grafo Turbo
├── model/
│   └── vocab.txt                    <-- Dicionário de tokens
├── reference/
│   └── referencia_voz.wav           <-- Áudio base para clonagem
├── manifest.json                    <-- Versão, backend e lista de arquivos
├── metadata.json                    <-- Contrato ONNX (inputs/outputs) e shapes
└── validation.json                  <-- Relatório de integridade e smoke test
```

## Contrato ONNX (Interface)
O arquivo `f5_tts_transformer_core.onnx` segue estritamente este contrato:

| Entrada | Tipo | Shape | Descrição |
| :--- | :--- | :--- | :--- |
| `x` | float32 | `[1, duration, 100]` | Tensor latente (ruído) |
| `cond` | float32 | `[1, duration, 100]` | Condicionamento Mel |
| `text_ids` | int64 | `[1, text_len]` | IDs do texto alvo |
| `text_lengths` | int64 | `[1]` | Comprimento real do texto |
| `time_steps` | float32 | `[1]` | Passo de tempo da difusão |

**Saída:**
- `dx` (float32): Velocidade prevista para o próximo passo.

## Como Gerar
1. Abra `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` no Kaggle.
2. Certifique-se de que a **Internet** está ligada.
3. Execute todas as células.
4. Baixe o arquivo `.zip` gerado na raiz do diretório de trabalho.

## Regras de Engenharia
- **Isolamento:** Nunca altera os arquivos originais em `voices/`.
- **Sincronia:** O notebook gera automaticamente o script `.py` para garantir que a lógica de exportação esteja sempre atualizada.
- **Validação:** O pacote só é considerado "Pronto" se passar no teste de carga do ONNX Runtime incluído no script.
