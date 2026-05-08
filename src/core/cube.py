"""Modelo de domínio central para conversão do estado do cubo mágico.

Este módulo contém apenas regras puras do estado do cubo. Ele não deve
depender de câmera, OpenCV, NumPy ou qualquer detalhe de visão computacional.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias

FaceName: TypeAlias = str
ColorName: TypeAlias = str
FaceGrid: TypeAlias = list[list[ColorName]]
FacesState: TypeAlias = dict[FaceName, FaceGrid]
CenterMap: TypeAlias = dict[ColorName, FaceName]
RotationMap: TypeAlias = dict[FaceName, int]
AdjacencyMap: TypeAlias = dict[str, FaceName]


class CubeStateError(ValueError):
    """Exceção lançada quando o estado do cubo não pode ser validado ou convertido."""


class Cube:
    """Representa um cubo 3x3 lido e o converte para a notação Kociemba.

    Args:
        faces: Mapeamento com as seis faces do cubo (U, R, F, D, L, B), cada
            uma contendo uma grade 3x3 com nomes de cores.

    Levanta:
        CubeStateError: Se o estado informado estiver incompleto ou com
            estrutura inválida.
    """

    FACE_ORDER: ClassVar[tuple[FaceName, ...]] = ("U", "R", "F", "D", "L", "B")
    UNDEFINED_COLOR: ClassVar[str] = "Nenhum"
    ADJACENCIES: ClassVar[dict[FaceName, AdjacencyMap]] = {
        "U": {"top": "B", "right": "R", "bottom": "F", "left": "L"},
        "R": {"top": "U", "right": "B", "bottom": "D", "left": "F"},
        "F": {"top": "U", "right": "R", "bottom": "D", "left": "L"},
        "D": {"top": "F", "right": "R", "bottom": "B", "left": "L"},
        "L": {"top": "U", "right": "F", "bottom": "D", "left": "B"},
        "B": {"top": "U", "right": "L", "bottom": "D", "left": "R"},
    }

    def __init__(self, faces: Mapping[FaceName, Sequence[Sequence[ColorName]]]) -> None:
        self._faces: FacesState = self._copy_faces(faces)
        self.validar_faces()

    @property
    def faces(self) -> FacesState:
        """Retorna uma cópia defensiva do estado original das faces lidas."""

        return self._copy_state(self._faces)

    def validar_faces(self) -> None:
        """Valida se todas as faces obrigatórias existem como grades 3x3.

        Levanta:
            CubeStateError: Se uma face obrigatória estiver ausente, se alguma
                grade não for 3x3 ou se algum adesivo estiver marcado como
                ``Nenhum``.
        """

        missing_faces = [face for face in self.FACE_ORDER if face not in self._faces]
        if missing_faces:
            raise CubeStateError(
                f"Erro: faces faltando no JSON: {', '.join(missing_faces)}"
            )

        for face in self.FACE_ORDER:
            self._validate_grid(face, self._faces[face], reject_undefined=True)

    def mapa_centros(self) -> CenterMap:
        """Mapeia cada cor central para a letra da face correspondente.

        Retorna:
            Um dicionário no formato ``{"cor": "U"}``.

        Levanta:
            CubeStateError: Se duas faces tiverem a mesma cor central.
        """

        self.validar_faces()
        return self._build_center_map()

    def converter_faces(self) -> FacesState:
        """Converte nomes de cores para letras Kociemba usando as cores centrais.

        Retorna:
            Um novo dicionário de faces contendo apenas adesivos U, R, F, D, L e B.

        Levanta:
            CubeStateError: Se alguma cor não tiver referência em uma cor central.
        """

        return self._convert_faces(self.mapa_centros())

    def orientar_faces(self) -> tuple[FacesState, RotationMap]:
        """Orienta as faces convertidas conforme as regras legadas de adjacência.

        Retorna:
            Uma tupla com o dicionário de faces orientadas e um mapeamento com
            as rotações horárias aplicadas a cada face.

        Levanta:
            CubeStateError: Se uma ou mais faces não puderem ser orientadas.
        """

        converted_faces = self.converter_faces()
        oriented_faces, rotations, problems = self._orient_faces(converted_faces)

        if problems:
            raise CubeStateError(
                "Erro: orientacao inconsistente nas faces: "
                + ", ".join(problems)
            )

        return oriented_faces, rotations

    def get_kociemba_string(self) -> str:
        """Retorna a string final do estado do cubo na ordem Kociemba URFDLB.

        Retorna:
            Uma string de 54 caracteres ordenada como U, R, F, D, L, B.

        Levanta:
            CubeStateError: Se o estado do cubo não puder ser validado,
                convertido ou orientado.
        """

        oriented_faces, _rotations = self.orientar_faces()
        return self._state_to_string(oriented_faces)

    @classmethod
    def _copy_faces(
        cls, faces: Mapping[FaceName, Sequence[Sequence[ColorName]]]
    ) -> FacesState:
        """Copia um mapeamento bruto para o formato mutável do estado privado."""

        if not isinstance(faces, Mapping):
            raise CubeStateError("Erro: o estado do cubo deve ser um dicionario")

        copied_faces: FacesState = {}
        for face, grid in faces.items():
            if not isinstance(face, str):
                raise CubeStateError("Erro: nomes de faces devem ser strings")
            copied_faces[face] = cls._copy_grid(face, grid)

        return copied_faces

    @classmethod
    def _copy_state(cls, faces: Mapping[FaceName, FaceGrid]) -> FacesState:
        """Retorna uma cópia defensiva de um estado de faces já normalizado."""

        return {face: cls._copy_grid(face, grid) for face, grid in faces.items()}

    @classmethod
    def _copy_grid(
        cls, face: FaceName, grid: Sequence[Sequence[ColorName]]
    ) -> FaceGrid:
        """Copia e valida a estrutura básica 3x3 de uma grade de face."""

        if not cls._is_sequence(grid) or len(grid) != 3:
            raise CubeStateError(f"Erro: face {face} nao tem grade 3x3")

        copied_grid: FaceGrid = []
        for row in grid:
            if not cls._is_sequence(row) or len(row) != 3:
                raise CubeStateError(f"Erro: face {face} nao tem grade 3x3")

            copied_row: list[ColorName] = []
            for color in row:
                if not isinstance(color, str):
                    raise CubeStateError(
                        f"Erro: face {face} contem valor de cor invalido"
                    )
                copied_row.append(color)

            copied_grid.append(copied_row)

        return copied_grid

    @staticmethod
    def _is_sequence(value: object) -> bool:
        """Verifica sequências não textuais usadas como grades de faces."""

        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        )

    @classmethod
    def _validate_grid(
        cls, face: FaceName, grid: FaceGrid, *, reject_undefined: bool
    ) -> None:
        """Valida uma grade normalizada e opcionalmente rejeita cores indefinidas."""

        if len(grid) != 3:
            raise CubeStateError(f"Erro: face {face} nao tem grade 3x3")

        for row in grid:
            if len(row) != 3:
                raise CubeStateError(f"Erro: face {face} nao tem grade 3x3")

            for color in row:
                if reject_undefined and color == cls.UNDEFINED_COLOR:
                    raise CubeStateError(
                        "Erro: ha cores indefinidas (Nenhum) no estado do cubo"
                    )

    def _build_center_map(self) -> CenterMap:
        """Monta o mapa de cor para face a partir dos adesivos centrais."""

        color_to_face: CenterMap = {}
        for face in self.FACE_ORDER:
            center_color = self._faces[face][1][1]
            if center_color in color_to_face:
                raise CubeStateError(
                    "Erro: cor de centro repetida: "
                    f"{center_color} nas faces {color_to_face[center_color]} e {face}"
                )
            color_to_face[center_color] = face

        return color_to_face

    def _convert_faces(self, color_to_face: CenterMap) -> FacesState:
        """Converte todos os adesivos de nomes de cores para letras de faces."""

        converted_faces: FacesState = {}

        for face in self.FACE_ORDER:
            converted_grid: FaceGrid = []

            for row in self._faces[face]:
                converted_row: list[ColorName] = []

                for color in row:
                    if color not in color_to_face:
                        raise CubeStateError(
                            f"Erro: cor sem referencia de centro: {color}"
                        )
                    converted_row.append(color_to_face[color])

                converted_grid.append(converted_row)

            converted_faces[face] = converted_grid

        return converted_faces

    @classmethod
    def _orient_faces(
        cls, converted_faces: Mapping[FaceName, FaceGrid]
    ) -> tuple[FacesState, RotationMap, list[FaceName]]:
        """Aplica as regras de orientação a cada face convertida."""

        oriented_faces: FacesState = {}
        rotations: RotationMap = {}
        problems: list[FaceName] = []

        for face in cls.FACE_ORDER:
            grid = cls._copy_grid(face, converted_faces[face])
            oriented_grid, rotation = cls._orient_face(face, grid)

            if oriented_grid is None or rotation is None:
                problems.append(face)
                oriented_faces[face] = cls._copy_grid(face, converted_faces[face])
            else:
                oriented_faces[face] = oriented_grid
                rotations[face] = rotation

        return oriented_faces, rotations, problems

    @classmethod
    def _orient_face(
        cls, face: FaceName, grid: FaceGrid
    ) -> tuple[FaceGrid | None, int | None]:
        """Rotaciona uma face até os adesivos laterais baterem com as adjacências."""

        expected = cls.ADJACENCIES[face]

        for rotation in range(4):
            if (
                grid[0][1] == expected["top"]
                and grid[1][2] == expected["right"]
                and grid[2][1] == expected["bottom"]
                and grid[1][0] == expected["left"]
            ):
                return grid, rotation

            grid = cls._rotate_face_clockwise(grid)

        return None, None

    @staticmethod
    def _rotate_face_clockwise(grid: FaceGrid) -> FaceGrid:
        """Rotaciona uma face 3x3 no sentido horário."""

        return [list(row) for row in zip(*grid[::-1])]

    @classmethod
    def _state_to_string(cls, converted_faces: Mapping[FaceName, FaceGrid]) -> str:
        """Serializa faces convertidas na ordem Kociemba URFDLB."""

        stickers: list[ColorName] = []

        for face in cls.FACE_ORDER:
            for row in converted_faces[face]:
                stickers.extend(row)

        return "".join(stickers)


__all__ = ["Cube", "CubeStateError"]