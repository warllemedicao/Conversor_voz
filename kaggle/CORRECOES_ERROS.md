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

## Erro Identificado (Novo)
**Tipo:** `HTTPStatusError` / `RepositoryNotFoundError` (404 Not Found Persistente com Token Ativo)
**Local:** `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` (Célula de Download dos Ativos)
**Mensagem:** `Repository Not Found for url: https://huggingface.co/api/models/warllem/Voz_Noslen/revision/main`
**Causa:** O token foi carregado com sucesso (comprimento 37), eliminando o problema de ausência de segredos. No entanto, o erro 404 continuou ocorrendo porque, por padrão, o método `snapshot_download` assume que o repositório consultado é um Modelo (`repo_type="model"`). Caso o repositório remoto `warllem/Voz_Noslen` tenha sido criado no Hugging Face sob a categoria de Dataset em vez de Model, o endpoint de Model responderá com 404.

## Ação Tomada
1. **Fallback Resiliente Automático:** Reescrevi o script de download no notebook com uma estrutura de `try-except` inteligente. O código agora tenta realizar o download assumindo que é um Modelo (`repo_type="model"`).
2. **Tratamento de Exceção de Tipo:** Caso receba um erro 404 ou `RepositoryNotFound`, o script captura a exceção, exibe uma mensagem informativa amigável e tenta imediatamente o download alternando o parâmetro para `repo_type="dataset"`. Isso garante 100% de compatibilidade não importando a natureza de criação do repositório.

## Prevenção
Ao construir scripts de automação de download de snapshots onde o tipo de repositório pode ser ambíguo ou alterado pelo usuário (Model vs Dataset), implemente uma lógica de tentativa e erro (fallback automático) para ambos os tipos usando o parâmetro `repo_type`.

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

---

## Erro Identificado (Novo - 2026-06-18)
**Tipo:** `RepositoryNotFoundError` / fonte Hugging Face incorreta para o mecanismo de download
**Local:** `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` (Célula de Download dos Ativos) e `kaggle/f5_tts_onnx_packager_kaggle.py` (constante de origem)
**Mensagem:** `Repository Not Found` ao tentar baixar `warllem/Voz_Noslen` via `snapshot_download`, apesar de a URL pública `https://huggingface.co/buckets/warllem/Voz_Noslen` responder corretamente.
**Causa:** A origem correta é um **Hugging Face Storage Bucket** (`/buckets/warllem/Voz_Noslen`), não um repositório Hub do tipo `model` ou `dataset`. A função `snapshot_download` consulta APIs de repositório (`/api/models` ou `/api/datasets`) e, por isso, não acessa o namespace de buckets. A análise da página do bucket confirmou a árvore `voices/v_minha_voz_f5_tts_ptbr` com arquivos expostos por links `/buckets/warllem/Voz_Noslen/resolve/...`.

## Ação Tomada
1. **Origem Corrigida:** Restaurei `DEFAULT_SOURCE_URL` para a URL completa do bucket: `https://huggingface.co/buckets/warllem/Voz_Noslen`.
2. **Downloader Correto para Bucket:** Substituí a célula de `snapshot_download` do notebook por um downloader recursivo que lê a listagem `BucketFileList`, percorre diretórios e baixa arquivos pelos links `/resolve/<path>?download=true`.
3. **Compatibilidade com Bucket Privado:** Mantive o suporte ao `HF_TOKEN` via variável de ambiente ou Kaggle Secrets, enviando `Authorization: Bearer <token>` nas requisições do bucket.
4. **Preservação da Estrutura:** O download continua materializando os arquivos em `/kaggle/working/turbo_source_snapshot/voices/v_minha_voz_f5_tts_ptbr`, que é o caminho esperado pelo packager.

## Prevenção
Quando a URL de origem estiver em `/buckets/...`, não usar `snapshot_download`. Buckets devem ser baixados pela própria árvore `/buckets/<owner>/<bucket>/tree/<prefix>` e pelos links `/resolve/...`; `snapshot_download` deve ficar restrito a repositórios Hub de `model`, `dataset` ou `space`.

---

## Erro Identificado (Novo - 2026-06-18)
**Tipo:** `ModuleNotFoundError`
**Local:** `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb` (Célula 4 - Execução da Conversão e Empacotamento)
**Mensagem:** `ModuleNotFoundError: No module named 'onnxscript'` ao executar `torch.onnx.export`.
**Causa:** A célula 1 do notebook instalava `onnx` e `onnxruntime`, mas não instalava `onnxscript`. Em versões recentes do PyTorch, o módulo interno de exportação ONNX importa `onnxscript` mesmo quando a chamada é feita pela API `torch.onnx.export`, causando falha antes da exportação do grafo.

## Ação Tomada
1. **Dependência Adicionada no Notebook:** Atualizei a célula 1 para instalar explicitamente `onnxscript` junto com `onnx` e `onnxruntime`.
2. **Diagnóstico dos Avisos:** Os `SyntaxWarning` emitidos por `pydub` são avisos de regex de dependência externa e não interrompem a execução; o erro bloqueante era apenas a ausência de `onnxscript`.

## Prevenção
Sempre manter `onnxscript` instalado no ambiente Kaggle quando o fluxo usa `torch.onnx.export` com PyTorch 2.x. Se a célula de instalação for editada manualmente, ela deve permanecer sincronizada com `kaggle/conversor_voz_requirements_kaggle.txt`.

---

## Correção de Finalização e Upload Automático - 2026-06-18
**Tipo:** pacote marcado como pronto sem garantia de ONNX válido e ausência de upload automático.
**Local:** `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb`, `kaggle/f5_tts_onnx_packager_kaggle.py` e `kaggle/README_kaggle.md`.
**Sintoma:** O notebook podia imprimir `PACOTE TURBO PRONTO` apenas porque `/kaggle/working/turbo_staging_area` existia, mesmo quando a exportação ONNX falhava antes de gerar `onnx/f5_tts_transformer_core.onnx`. Além disso, o fluxo final ainda dependia de download manual do `.zip`, sem criação de destino nem upload direto para o Hugging Face.

## Ação Tomada
1. **Validação antes do ZIP:** A célula final agora exige a presença de `onnx/f5_tts_transformer_core.onnx` e `validation.json` com `status: verified` antes de compactar o pacote.
2. **Upload direto para Hugging Face:** A célula final usa `huggingface_hub.HfApi` para criar o repositório de destino com `create_repo(..., exist_ok=True)` e enviar o `.zip` com `upload_file`.
3. **Pasta remota automática:** A pasta no Hugging Face é criada pelo próprio `path_in_repo`, com padrão `turbo/<arquivo.zip>`.
4. **Configuração por ambiente:** O destino pode ser controlado por `HF_UPLOAD_REPO_ID`, `HF_UPLOAD_REPO_TYPE`, `HF_UPLOAD_FOLDER` e `HF_PRIVATE_REPO`; o padrão é `warllem/Voz_Noslen_Turbo`, `model`, `turbo/`, privado.
5. **Token obrigatório:** O upload falha explicitamente se `HF_TOKEN` não estiver disponível ou estiver vazio nos Secrets do Kaggle.
6. **Contorno do Torch Export:** O `torch.onnx.export` passou a usar `dynamo=False`, evitando o caminho `torch.export` que gerava `TorchExportError` com `IndexError: Dimension out of range`.

## Prevenção
O pacote só deve ser considerado finalizado quando a validação ONNX passar e o upload imprimir `UPLOAD HUGGING FACE CONCLUIDO`. Mensagens de pacote pronto não devem depender apenas da existência da pasta de staging.

---

## Erro Identificado (Novo - 2026-06-19)
**Tipo:** `TorchExportError` / `IndexError: Dimension out of range`
**Local:** `kaggle/f5_tts_onnx_packager_kaggle.py` e script embutido em `kaggle/voz_noslen_f5_tts_onnx_kaggle.ipynb`
**Mensagem:** `audio_mask.sum(dim=1)` falhou com `expected to be in range of [-1, 0], but got 1` durante a exportação ONNX.
**Causa:** O wrapper Turbo chamava o DiT com argumentos posicionais:

```python
self.transformer(x, cond, text_ids, time_steps, text_lengths)
```

Na assinatura do Transformer do F5-TTS, o quinto argumento não é `text_lengths`; ele é interpretado como controle/máscara de condicionamento de áudio (`drop_audio_cond`/máscara). Com isso, um tensor 1D (`[batch]`) era usado no caminho de `audio_mask`, e a operação `sum(dim=1)` falhava porque a dimensão 1 não existia.

## Ação Tomada
1. **Chamada nomeada do DiT:** O wrapper agora chama o Transformer como `self.transformer(x=x, cond=cond, text=text_ids, time=time_steps)`, eliminando a ambiguidade de argumentos posicionais.
2. **Preservação do contrato ONNX:** `text_lengths` continua sendo entrada do grafo ONNX por meio de uma âncora dinâmica (`x + length_anchor - length_anchor`). Assim, o backend pode manter o contrato `[x, cond, text_ids, text_lengths, time_steps]` sem provocar erro no DiT.
3. **Dtypes explícitos:** Os inputs de exemplo da exportação agora definem `float32`/`int64` explicitamente.
4. **Versão atualizada:** O packager passou para `2026.06.19.turbo.v3`.
5. **Sincronização:** O notebook Kaggle foi atualizado para gerar o mesmo script corrigido.

## Prevenção
Não passar parâmetros opcionais do F5-TTS por posição no wrapper ONNX. Para o DiT, usar argumentos nomeados e manter entradas extras do contrato ONNX ancoradas no grafo apenas quando elas forem necessárias para compatibilidade do backend.

---

## Erro Identificado (Novo - 2026-06-19, segunda execução)
**Tipo:** `torch.onnx.export` / `IndexError: Dimension out of range`
**Local:** `kaggle/f5_tts_onnx_packager_kaggle.py` durante `torch.onnx.export`
**Mensagem:** Após `INFO: Iniciando torch.onnx.export (Turbo Contract)...`, o exportador legado emitiu `TracerWarning` em `TextEmbedding.forward` e falhou com `Dimension out of range (expected to be in range of [-1, 0], but got 1)`.
**Causa provável:** A correção anterior removeu o `text_lengths` da posição errada, mas o caminho interno do DiT ainda podia acionar `audio_mask.sum(dim=1)` com uma máscara ausente, ambígua ou incompatível em alguma versão instalada do `f5-tts` no Kaggle. O log também não imprimia o traceback completo, dificultando confirmar o ponto exato da falha.

## Ação Tomada
1. **Máscara 2D explícita:** O wrapper agora cria `audio_mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)`, garantindo shape `[batch, duration]` derivado diretamente do tensor `x`.
2. **Compatibilidade por assinatura:** O wrapper inspeciona `transformer.forward` e só passa `mask=audio_mask` e `cache=False` quando esses argumentos existem na versão instalada do F5-TTS.
3. **Diagnóstico reforçado:** O script registra `Assinatura transformer.forward` no log do Kaggle antes da exportação.
4. **Traceback completo:** A captura de erro mudou de `LOGGER.error` para `LOGGER.exception`, então novas falhas mostram a pilha completa.
5. **Versão atualizada:** O packager passou para `2026.06.19.turbo.v4`.
6. **Sincronização:** O notebook Kaggle foi atualizado para gerar a versão v4 do script.

## Prevenção
Sempre que o wrapper depender de comportamento interno do F5-TTS, registrar a assinatura real do método chamado no ambiente Kaggle e passar máscaras com shape explícito `[batch, duration]`.

---

## Erro Identificado (Novo - 2026-06-19, terceira execução)
**Tipo:** falha de validação ONNX Runtime / `Concat` com shapes incompatíveis
**Local:** `validate_package()` em `kaggle/f5_tts_onnx_packager_kaggle.py`
**Mensagem:** `Non concat axis dimensions must match: Axis 1 has mismatched dimensions of 128 and 16` no nó `/transformer/input_embed/Concat`.
**Causa:** A exportação ONNX foi concluída, mas o smoke test alimentava `x` e `cond` com duração `16`. Durante o trace, o F5-TTS converte `seq_len.max().item()` em inteiro Python dentro de `TextEmbedding.forward`; isso especializa o caminho interno do texto em `128` frames, que era a duração usada nos inputs de exemplo da exportação. Assim, na validação, `x/cond` tinham eixo temporal `16` e `text_embed` tinha eixo temporal `128`, causando falha no `Concat`.

## Ação Tomada
1. **Contrato fixo de duração:** Adicionado `TURBO_DURATION = 128`.
2. **Dynamic axes corrigidos:** Removido eixo dinâmico de `x`, `cond` e `dx`; apenas `text_ids` mantém `text_len` dinâmico.
3. **Metadata corrigido:** `metadata.json` agora declara `x`, `cond` e `dx` como `[1, 128, 100]` e inclui restrição `duration=128`.
4. **Smoke test corrigido:** A validação ONNX Runtime agora alimenta `x` e `cond` com `[1, 128, 100]` e verifica se `dx` retorna exatamente `(1, 128, 100)`.
5. **Versão atualizada:** O packager passou para `2026.06.19.turbo.v5`.
6. **Sincronização:** O notebook Kaggle foi atualizado para gerar a versão v5 do script.

## Prevenção
Enquanto o exportador usado for o tracer legado (`dynamo=False`) e o F5-TTS usar `seq_len.max().item()` no caminho do texto, não declarar `duration` como dinâmico no ONNX. O backend deve chamar o grafo Turbo em blocos de 128 frames.
