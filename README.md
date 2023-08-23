# LEMI - 3W

## Introdução

Esse trabalho buscar realizar classificação multiclasse de anomalias, utilizando técnicas de MLOps com modelos LSTM. Aplicado ao problema da [dataset 3W](https://github.com/petrobras/3W) da Petrobras.

Desenvolvido por Marcus Carr sob orientação de Carlos Diaz. Trabalhos ligados ao Laboratório de Escoamento Multifásicos Industriais (LEMI-EESC-USP).

Pesquisa desenvolvida com utilização dos recursos computacionais do Centro de Ciências Matemáticas Aplicadas à Indústria (CeMEAI), financiados pela FAPESP (proc. 2013/07375-0).

---

## Descrição do ambiente

*constants*: constantes utilizadas por outros módulos.

*pipeline*: gerencia os módulos da Pipeline utilizando TensorFlow Extended.

*raw_data_manager*: aquisição e inspecção dos dados da dataset 3W.

*tests*: pasta de testes para *raw_data_manager*.

Gerenciamento de dependências feito usando *poetry*. Formatação usando *black*.
