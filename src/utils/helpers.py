"""Funcoes auxiliares puras usadas em visao e orientacao do cubo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeVar

import numpy as np

T = TypeVar("T")
Matrix: TypeAlias = Sequence[Sequence[T]]


def ordenar_pontos(pts: np.ndarray) -> np.ndarray:
    """Ordena quatro pontos como topo-esquerda, topo-direita, baixo-direita e baixo-esquerda.

    Args:
        pts: Array contendo quatro pontos 2D em qualquer ordem. Aceita formatos
            comuns do OpenCV, como ``(4, 1, 2)`` ou ``(4, 2)``.

    Returns:
        Um array ``float32`` com shape ``(4, 2)`` na ordem esperada pelo
        ``cv2.getPerspectiveTransform``.
    """

    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(point_sums)]
    ordered[2] = points[np.argmax(point_sums)]
    ordered[1] = points[np.argmin(point_diffs)]
    ordered[3] = points[np.argmax(point_diffs)]

    return ordered


def rotacionar_cw(matriz: Matrix[T]) -> list[list[T]]:
    """Rotaciona uma matriz no sentido horario."""

    rows = [list(row) for row in matriz]
    return [list(row) for row in zip(*rows[::-1])]


def rotacionar_ccw(matriz: Matrix[T]) -> list[list[T]]:
    """Rotaciona uma matriz no sentido anti-horario."""

    rows = [list(row) for row in matriz]
    return [list(row) for row in zip(*rows)][::-1]


def rotacionar_180(matriz: Matrix[T]) -> list[list[T]]:
    """Rotaciona uma matriz em 180 graus."""

    rows = [list(row) for row in matriz]
    return [list(reversed(row)) for row in reversed(rows)]


__all__ = ["ordenar_pontos", "rotacionar_180", "rotacionar_ccw", "rotacionar_cw"]
