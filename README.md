# LEMI - 3W

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Introdução

Esse trabalho buscar realizar classificação multiclasse de anomalias, utilizando técnicas de MLOps com modelos LSTM. Aplicado ao problema da [dataset 3W](https://github.com/petrobras/3W) da Petrobras.

Desenvolvido por Marcus Carr sob orientação de Carlos Diaz. Trabalhos ligados ao Laboratório de Escoamento Multifásicos Industriais (LEMI-EESC-USP).

---

## Descrição do ambiente

*constants*: constantes utilizadas por outros módulos.

*raw_data_manager*: aquisição e inspecção dos dados da dataset 3W.

*data_exploration*: ferramentas para visualização e aquisição de métricas.

*data_preparation*: aplica transformações nos dados.

*experiments*: notebooks com experimentos de modelagem.

*tests*: pasta de testes para *raw_data_manager*.

Gerenciamento de dependências feito usando *[poetry](https://python-poetry.org/)*. Formatação usando *black*.
