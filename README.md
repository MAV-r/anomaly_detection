# MLOps Project
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![version](https://img.shields.io/badge/version-0.1.0-blue)

This is a simple project for MLOps course @ HSE MLHLS. 

# Используемые инструменты:

1. Poetry. Все зависимости добавлены в pyproject.toml.
2. Pre-commit. В проекте реализовано применение Pre-commit в составе хуков: black, isort, flake8, pylint и prettier.
3. DVC. Управление версиями данных и их хранение производится через DVC на удаленном диске google-drive. 
Скачивание необходимых данных добавлено в процесс train и infer. 
4. Hydra. Для управления конфигурациями используется hydra. Конфигурации представлены в папке сonfigs.
5. Logging. Логирование осуществляется через mlflow.

