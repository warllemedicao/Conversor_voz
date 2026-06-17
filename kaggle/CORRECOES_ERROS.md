# Relatório de Correções de Erros - 2026-06-17

## Erro Identificado
**Tipo:** `SyntaxError`
**Local:** `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` (Script embutido)
**Mensagem:** `SyntaxError: closing parenthesis ']' does not match opening parenthesis '('` na linha 424.
**Causa:** Escapamento incorreto de aspas dentro de uma string de lista no notebook. A linha original continha `re.findall(r"href=["']([^"']+)["']", ...)` onde as aspas duplas internas encerraram prematuramente a string da linha no Python do notebook, resultando em um código inválido sendo escrito no arquivo `.py` final.

## Ação Tomada
1.  **Correção do Escapamento:** O regex foi alterado para usar um nível mais profundo de backslashes (`\\\\\\\"` e `\\\\\\\'`) para garantir que, ao ser escrito pelo notebook no arquivo `.py`, ele resulte no código Python válido: `re.findall(r'href=["\']([^"\']+)["\']', response.text)`.
2.  **Sincronização:** O arquivo `kaggle/f5_tts_onnx_packager_kaggle.py` foi verificado para garantir que a versão "master" também esteja correta.
3.  **Reforço:** Atualizei o notebook com a nova estratégia de escapamento para evitar que o erro se repita em futuras gerações do script via notebook.

## Prevenção
Para evitar que este erro se repita, ao embutir scripts Python em strings de listas de notebooks, deve-se:
*   Usar ferramentas de automação para gerar o JSON do notebook a partir do arquivo `.py` em vez de edição manual do JSON.
*   Validar o script `.py` gerado localmente antes de publicar o notebook.

---

## Erro Identificado (Novo)
**Tipo:** `TorchExportError` / `GuardOnDataDependentSymNode`
**Local:** `kaggle/f5_tts_onnx_packager_kaggle.py` (Exportação ONNX)
**Mensagem:** `Could not guard on data-dependent expression u0 + 6 < 7`.
**Causa:** O novo exportador ONNX do PyTorch (baseado em Dynamo/torch.export) não conseguia validar se o comprimento da sequência de áudio era válido para as operações de convolução interna do modelo `Vocos`. Isso ocorre devido ao uso de formas simbólicas (dynamic shapes) que dependem de cálculos em tempo de execução (`text_ids.shape[1]` e `speed`).

## Ação Tomada
1.  **Hints para o Exportador:** Adicionei chamadas `torch._check()` dentro do método `forward` do wrapper. Estas chamadas servem como "garantias" estáticas para o exportador simbólico, confirmando que o comprimento da sequência sempre satisfará as restrições matemáticas do modelo (`mel.shape[2] >= 32`). Isso resolve o erro de guarda em expressões dependentes de dados durante o export.
2.  **Sincronização:** Atualizei tanto o script `.py` quanto a versão embutida no notebook `.ipynb`.

## Prevenção
Sempre que utilizar o novo exportador ONNX do PyTorch com modelos que possuam lógica condicional ou restrições de tamanho (como convoluções com kernels específicos), utilize `torch._check` para fornecer metadados sobre as dimensões dinâmicas.

---

## Resolução Final - Arquitetura Turbo (v2026.06.17)
**Status:** Implementado e Sincronizado.
**Ação:** O projeto foi estabilizado na **Arquitetura Turbo**. Esta arquitetura separa o núcleo do Transformer (exportado em ONNX) do loop de inferência ODE (mantido em Python).
**Benefícios:**
1.  **Eliminação de Erros de Tipo:** Removeu-se a necessidade de `torch._check` e condicionais complexas que falhavam no Dynamo.
2.  **Transparência:** Adição de `manifest.json`, `metadata.json` e `validation.json` para garantir que o backend Cloud Run receba todas as informações necessárias de contrato e shapes.
3.  **Isolamento:** O fluxo de exportação agora opera em uma área de staging isolada, garantindo 0% de risco aos arquivos originais do projeto "Mainha".

## Prevenção Permanente
Para novos modelos, o fluxo `f5_tts_onnx_packager_kaggle.py` deve ser seguido como o padrão ouro para exportação ONNX em CPU.
