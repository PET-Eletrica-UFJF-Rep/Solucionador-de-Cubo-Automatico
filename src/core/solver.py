"""Resolvedor do cubo usando a biblioteca kociemba."""

from __future__ import annotations

from typing import ClassVar

import kociemba

from config.settings import FACE_NOMES


class SolverError(RuntimeError):
    """Excecao lancada quando o estado do cubo nao pode ser resolvido."""


class Solver:
    """Encapsula a integracao com o resolvedor Kociemba.

    Esta classe assume que a validacao estrutural e a conversao do estado do
    cubo ja foram feitas pela classe ``Cube``. Sua responsabilidade e apenas
    enviar a string Kociemba para a biblioteca externa e formatar os movimentos
    retornados para exibicao.
    """

    DIRECTION_BY_MODIFIER: ClassVar[dict[str, str]] = {
        "'": "90 graus no sentido ANTI-HORARIO",
        "2": "180 graus (Meia volta completa)",
        "": "90 graus no sentido HORARIO",
    }

    def solve(self, cube_state_string: str) -> list[str]:
        """Resolve um estado de cubo na notacao Kociemba.

        Args:
            cube_state_string: String de 54 caracteres na ordem Kociemba
                ``URFDLB``. A validacao detalhada desse estado pertence a
                ``Cube`` e a propria biblioteca ``kociemba``.

        Returns:
            Lista de movimentos calculados pela biblioteca, por exemplo
            ``["U", "R'", "F2"]``. Se a biblioteca devolver uma solucao vazia,
            retorna uma lista vazia.

        Raises:
            SolverError: Se a biblioteca ``kociemba`` rejeitar o estado como
                invalido.
        """

        try:
            solution = kociemba.solve(cube_state_string)
        except ValueError as exc:
            raise SolverError(
                "Erro: estado invalido para o kociemba. "
                "Verifique se a string usa a ordem U R F D L B corretamente."
            ) from exc

        return solution.split()

    def format_solution(self, moves: list[str]) -> str:
        """Formata uma lista de movimentos como instrucoes passo a passo.

        Args:
            moves: Lista de movimentos em notacao Kociemba, como ``"U"``,
                ``"R'"`` ou ``"F2"``.

        Returns:
            Uma string multilinhas pronta para ser mostrada por uma interface
            grafica ou por um terminal.
        """

        lines: list[str] = [
            "",
            "=" * 50,
            " INSTRUCOES PASSO A PASSO PARA RESOLVER O CUBO ",
            "=" * 50,
            "IMPORTANTE: Segure o cubo com a face FRENTE (F) virada para voce",
            "            e a face TOPO (U) apontando para cima (para o teto).",
            "",
        ]

        if not moves:
            lines.append("Nenhum movimento necessario: o cubo ja esta resolvido.")
        else:
            for index, move in enumerate(moves, start=1):
                face_letter = move[0]
                modifier = move[1:] if len(move) > 1 else ""
                face_name = FACE_NOMES.get(face_letter, face_letter)
                direction = self.DIRECTION_BY_MODIFIER.get(
                    modifier, f"movimento {modifier}"
                )

                lines.append(
                    f"Passo {index:02d}: "
                    f"Gire a face {face_name.upper():<8} -> {direction}"
                )

        lines.extend(
            [
                "",
                "Parabens! O cubo deve estar resolvido!",
                "=" * 50,
            ]
        )

        return "\n".join(lines)

    def translate_solution(self, moves: list[str]) -> str:
        """Alias semantico para ``format_solution``.

        Args:
            moves: Lista de movimentos em notacao Kociemba.

        Returns:
            A solucao formatada como string multilinhas.
        """

        return self.format_solution(moves)


__all__ = ["Solver", "SolverError"]
