"""Funções utilitárias para leitura e escrita de arquivos JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileAccessError(OSError):
    """Exceção lançada quando um arquivo JSON não pode ser acessado corretamente."""


def load_json(filepath: str) -> dict[str, Any]:
    """Carrega um arquivo JSON e retorna seu conteúdo como dicionário.

    Args:
        filepath: Caminho do arquivo JSON que será lido.

    Retorna:
        O conteúdo do arquivo JSON como um dicionário.

    Levanta:
        FileAccessError: Se o arquivo não existir, estiver corrompido, não puder
            ser lido ou não contiver um objeto JSON na raiz.
    """

    path = Path(filepath)

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise FileAccessError(f"Arquivo JSON não encontrado: {filepath}") from exc
    except json.JSONDecodeError as exc:
        raise FileAccessError(f"Arquivo JSON corrompido ou inválido: {filepath}") from exc
    except OSError as exc:
        raise FileAccessError(f"Não foi possível ler o arquivo JSON: {filepath}") from exc

    if not isinstance(data, dict):
        raise FileAccessError(f"O arquivo JSON deve conter um objeto na raiz: {filepath}")

    return data


def save_json(filepath: str, data: dict[str, Any]) -> None:
    """Salva um dicionário em um arquivo JSON.

    Args:
        filepath: Caminho do arquivo JSON que será criado ou sobrescrito.
        data: Dicionário que será serializado em JSON.

    Levanta:
        FileAccessError: Se o arquivo não puder ser escrito ou se os dados não
            puderem ser serializados como JSON.
    """

    path = Path(filepath)

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as exc:
        raise FileAccessError(f"Não foi possível salvar o arquivo JSON: {filepath}") from exc


__all__ = ["FileAccessError", "load_json", "save_json"]
