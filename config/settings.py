"""Configurações globais e constantes estáticas do projeto."""

from __future__ import annotations

from typing import Final, TypeAlias

ColorBGR: TypeAlias = tuple[int, int, int]

CORES_VISUAIS: Final[dict[str, ColorBGR]] = {
    "vermelho": (0, 0, 255),
    "verde": (0, 255, 0),
    "azul": (255, 0, 0),
    "amarelo": (0, 255, 255),
    "laranja": (0, 165, 255),
    "branco": (255, 255, 255),
}

FACE_ORDEM: Final[tuple[str, ...]] = ("U", "R", "F", "D", "L", "B")

FACE_NOMES: Final[dict[str, str]] = {
    "U": "Topo",
    "R": "Direita",
    "F": "Frente",
    "D": "Baixo",
    "L": "Esquerda",
    "B": "Tras",
}

JANELA_DETECCAO: Final[str] = "Deteccao Automatica"
JANELA_FACE: Final[str] = "Face Lida (Grade 3x3)"

TAMANHO_QUADRADO: Final[int] = 100
MARGEM_LEITURA: Final[int] = 30
LIMIAR_LEITURA: Final[float] = 0.15

TAMANHO_ROI: Final[int] = 300

# Alias temporário para manter compatibilidade com o nome usado no script legado.
cores_visuais: Final[dict[str, ColorBGR]] = CORES_VISUAIS

__all__ = [
    "CORES_VISUAIS",
    "ColorBGR",
    "FACE_NOMES",
    "FACE_ORDEM",
    "JANELA_DETECCAO",
    "JANELA_FACE",
    "LIMIAR_LEITURA",
    "MARGEM_LEITURA",
    "TAMANHO_QUADRADO",
    "TAMANHO_ROI",
    "cores_visuais",
]
