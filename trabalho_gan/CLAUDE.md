# Instruções do projeto

## Commits

Todos os commits devem seguir o padrão internacional [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>[escopo opcional]: <descrição>
```

Tipos principais: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`.

Exemplos:
- `feat(dataset): adiciona subconjunto configurável do Fashion-MNIST`
- `fix(train): corrige checkpoint salvo com shape errado`
- `docs: atualiza README com instruções da cGAN`

Mensagens em português, descrição no imperativo, sem ponto final.

## Testes

Sempre escrever os testes antes de implementar a funcionalidade (TDD) — implementar só depois que os testes estiverem definidos.
