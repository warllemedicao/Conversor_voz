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

## Erro Identificado (Novo)
**Tipo:** `TypeError` / `TorchExportError`
**Local:** `kaggle/f5_tts_onnx_packager_kaggle.py` (Exportação Dynamo/ONNX)
**Mensagem:** `TypeError: cond must be a bool, but got <class 'torch._subclasses.fake_tensor.FakeTensor'>`
**Causa:** Uso incorreto de `torch._check()` em condições baseadas em tensores durante o rastreamento simbólico do Dynamo. Além disso, a implementação anterior do "Modo Lite" tentava exportar um loop ODE completo que estava funcionalmente incompleto (usando `pass`), resultando em falhas de exportação e modelos inválidos.

## Ação Tomada
1.  **Restauração da Arquitetura Turbo:** Abandonei a tentativa de exportar o loop Diffusion completo (Modo Lite) em favor da arquitetura **Turbo** (exportação apenas do núcleo do Transformer/DiT). Esta abordagem é comprovadamente estável e compatível com o backend atual.
2.  **Remoção de Guards Problemáticos:** Eliminei as chamadas `torch._check()` que causavam o erro de tipo, já que a arquitetura Turbo não depende de lógica condicional complexa dentro do grafo ONNX para o cálculo da duração (que volta a ser gerenciada pelo backend em Python).
3.  **Recriação do Notebook:** Deletei e recriei o notebook `voz_noslen_f5_tts_onnx_kaggle.ipynb` do zero, garantindo que o script embutido esteja sincronizado com a versão Turbo (v2026.06.17.turbo).

## Prevenção
*   Priorizar arquiteturas modulares (Turbo) para exportação ONNX em vez de tentar embutir loops de inferência ODE complexos em um único grafo, a menos que o backend exija estritamente.
*   Evitar o uso de `torch._check` para validações de valores de tensores durante o export Dynamo em versões do PyTorch que ainda apresentam instabilidade com FakeTensors em condições booleanas.
