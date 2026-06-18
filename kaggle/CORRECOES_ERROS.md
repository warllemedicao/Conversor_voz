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
**Tipo:** `HTTPStatusError` / `RepositoryNotFoundError` (401 Unauthorized)
**Local:** `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` (Célula de Download dos Ativos)
**Mensagem:** `Client error '401 Unauthorized' for url 'https://huggingface.co/api/models/warllem/Voz_Noslen/revision/main'`
**Causa:** O repositório do modelo `warllem/Voz_Noslen` exige autenticação/token de acesso (por ser privado ou restrito), mas a chamada da função `snapshot_download` não estava fornecendo um token.

## Ação Tomada
1. **Integração com Kaggle Secrets:** Modifiquei a célula de download do notebook para obter o token de autenticação via variável de ambiente `HF_TOKEN` ou puxar diretamente do Kaggle Secrets (utilizando `UserSecretsClient().get_secret("HF_TOKEN")`).
2. **Parâmetro de Autenticação:** Repassei o token obtido como argumento `token=hf_token` na chamada do `snapshot_download` para assegurar o acesso autorizado aos ativos.

## Prevenção
Para repositórios privados ou controlados do Hugging Face executados no Kaggle, sempre implemente o fallback seguro para ler credenciais do Kaggle Secrets (`HF_TOKEN`), evitando falhas em downloads automatizados.

---

## Erro Identificado (Novo)
**Tipo:** `HTTPStatusError` / `RepositoryNotFoundError` (404 Not Found)
**Local:** `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` (Célula de Download dos Ativos)
**Mensagem:** `Client error '404 Not Found' for url 'https://huggingface.co/api/models/warllem/Voz_Noslen/revision/main'`
**Causa:** No ecossistema do Hugging Face Hub, se um token é enviado mas não possui permissão de acesso ao repositório privado específico ou se o token está em branco/inválido, a API do Hugging Face responde intencionalmente com `404 Not Found` (em vez de 401) para ocultar a existência do repositório privado. No Kaggle, isso costuma ocorrer quando os "Secrets" estão cadastrados na conta, mas o usuário esqueceu de marcar a caixinha de verificação para habilitar esse Segredo específico dentro do notebook atual em execução.

## Ação Tomada
1. **Mensagens de Diagnóstico Embutidas:** Inseri um bloco de validação visual e inteligência no notebook. Agora ele imprime se o token foi encontrado e o seu comprimento numérico real, sem expor os caracteres confidenciais.
2. **Alertas e Dicas Acionáveis:** Caso o token não seja carregado (retornando vazio ou gerando exceção devido a não-autorização de add-ons), o código exibe instruções passo a passo instruindo o usuário a acessar o menu `Add-ons` -> `Secrets` no painel lateral do Kaggle e ativar o botão de seleção do `HF_TOKEN` para este notebook específico.

## Prevenção
Sempre que utilizar segredos no Kaggle para autenticação de repositórios privados, inclua blocos de sanidade e instruções interativas de interface (`Add-ons -> Secrets`) para guiar a configuração correta do ambiente de runtime.

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
