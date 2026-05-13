"""Ponto de entrada da aplicacao do solucionador de cubo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import cv2

from config.settings import FACE_NOMES, FACE_ORDEM, JANELA_DETECCAO, JANELA_FACE
from src.core.cube import Cube, CubeStateError
from src.core.solver import Solver, SolverError
from src.utils.file_manager import FileAccessError, load_json
from src.utils.helpers import rotacionar_180, rotacionar_ccw, rotacionar_cw
from src.vision.camera import Camera, CameraConnectionError
from src.vision.scanner import FaceScanner, ScanFrameResult, ScannerCalibrationError

StepKind = Literal["capture", "return_u"]
FaceGrid = list[list[str]]
RotationFunction = Callable[[FaceGrid], FaceGrid]

CALIBRATION_FILE: Final[str] = "cores_cubo.json"
STABLE_FRAMES_REQUIRED: Final[int] = 10
ESC_KEY: Final[int] = 27
WINDOW_WIDTH: Final[int] = 1280
WINDOW_HEIGHT: Final[int] = 720

CAPTURE_HINTS: Final[dict[str, str]] = {
    "U": "(sua face inicial)",
    "F": "-> Tombe o cubo P/ TRAS 1x",
    "R": "-> Tombe o cubo P/ ESQUERDA",
    "D": "-> Tombe o cubo P/ TRAS 2x",
    "L": "-> Tombe o cubo P/ DIREITA",
    "B": "-> Tombe o cubo P/ FRENTE 1x",
}

FACE_ROTATIONS: Final[dict[str, RotationFunction]] = {
    "R": rotacionar_cw,
    "L": rotacionar_ccw,
    "B": rotacionar_180,
}

CUSTOM_ERRORS: Final[tuple[type[Exception], ...]] = (
    FileAccessError,
    CameraConnectionError,
    ScannerCalibrationError,
    CubeStateError,
    SolverError,
)


@dataclass(frozen=True, slots=True)
class CaptureStep:
    """Representa uma etapa da maquina de estados de captura."""

    kind: StepKind
    face: str | None = None


def build_capture_steps() -> list[CaptureStep]:
    """Monta a sequencia de leitura esperada para as seis faces do cubo."""

    steps = [CaptureStep("capture", "U")]
    for face in ("F", "R", "D", "L", "B"):
        steps.append(CaptureStep("return_u"))
        steps.append(CaptureStep("capture", face))

    return steps


def run() -> int:
    """Executa o fluxo principal da aplicacao."""

    captured_faces: dict[str, FaceGrid] = {}

    try:
        calibrations = load_json(CALIBRATION_FILE)
        scanner = FaceScanner(calibrations)
        steps = build_capture_steps()
        captured_faces = capture_faces(scanner, steps)

        if len(captured_faces) != len(FACE_ORDEM):
            print("Captura interrompida antes de ler as seis faces.")
            return 0

        cube = Cube(captured_faces)
        cube_state_string = cube.get_kociemba_string()

        solver = Solver()
        moves = solver.solve(cube_state_string)

        print("\nEstado do Cubo (Notacao Kociemba):")
        print(cube_state_string)
        print("\nSolucao Bruta:")
        print(" ".join(moves))
        print(solver.translate_solution(moves))
        return 0

    except CUSTOM_ERRORS as exc:
        print("\nNao foi possivel concluir a operacao.")
        print(str(exc))
        return 1
    except KeyboardInterrupt:
        print("\nOperacao interrompida pelo utilizador.")
        return 0
    finally:
        cv2.destroyAllWindows()


def capture_faces(scanner: FaceScanner, steps: list[CaptureStep]) -> dict[str, FaceGrid]:
    """Captura as seis faces do cubo usando camera e scanner."""

    captured_faces: dict[str, FaceGrid] = {}
    step_index = 0
    stable_count = 0
    last_signature: tuple[tuple[str, ...], ...] | tuple[str, str] | None = None
    top_center_color: str | None = None

    cv2.namedWindow(JANELA_DETECCAO, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JANELA_DETECCAO, WINDOW_WIDTH, WINDOW_HEIGHT)

    with Camera() as cam:
        while step_index < len(steps):
            frame = cam.get_frame()
            result = scanner.scan_frame(frame)
            current_step = steps[step_index]

            signature = get_stable_signature(result, current_step, top_center_color)
            if signature is None:
                stable_count = 0
                last_signature = None
            elif signature == last_signature:
                stable_count += 1
            else:
                stable_count = 1
                last_signature = signature

            if (
                stable_count >= STABLE_FRAMES_REQUIRED
                and current_step.kind == "capture"
                and current_step.face is not None
                and result.grid is not None
            ):
                face = current_step.face
                captured_faces[face] = apply_capture_rotation(face, result.grid)

                if face == "U":
                    top_center_color = captured_faces[face][1][1]

                print(f"Face capturada: {face}")
                step_index += 1
                stable_count = 0
                last_signature = None
                scanner.reset_tracking()
            elif stable_count >= STABLE_FRAMES_REQUIRED:
                step_index += 1
                stable_count = 0
                last_signature = None
                scanner.reset_tracking()

            if step_index < len(steps):
                current_step = steps[step_index]

            frame_to_show = (
                result.annotated_frame if result.annotated_frame is not None else frame
            )
            draw_hud(
                frame_to_show,
                current_step,
                captured_faces,
                stable_count,
                STABLE_FRAMES_REQUIRED,
            )
            show_windows(frame_to_show, result)

            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                break

    return captured_faces


def get_stable_signature(
    result: ScanFrameResult,
    step: CaptureStep,
    top_center_color: str | None,
) -> tuple[tuple[str, ...], ...] | tuple[str, str] | None:
    """Extrai a assinatura usada para confirmar que a leitura esta estavel."""

    if not result.has_valid_grid or result.grid is None:
        return None

    if step.kind == "return_u":
        if top_center_color is None or result.center_color != top_center_color:
            return None
        return ("return_u", result.center_color)

    if step.face == "U":
        return grid_signature(result.grid)

    if top_center_color is None or result.center_color == top_center_color:
        return None

    return grid_signature(result.grid)


def grid_signature(grid: FaceGrid) -> tuple[tuple[str, ...], ...]:
    """Converte uma grade mutavel numa assinatura hashable."""

    return tuple(tuple(row) for row in grid)


def apply_capture_rotation(face: str, grid: FaceGrid) -> FaceGrid:
    """Aplica a rotacao de orientacao legada antes de guardar a face."""

    rotation = FACE_ROTATIONS.get(face)
    if rotation is None:
        return [row.copy() for row in grid]

    return rotation(grid)


def draw_hud(
    frame,
    step: CaptureStep,
    captured_faces: dict[str, FaceGrid],
    stable_count: int,
    stable_frames_required: int,
) -> None:
    """Desenha as instrucoes da etapa atual no frame anotado."""

    text_color = (0, 255, 255)
    white = (255, 255, 255)

    cv2.putText(
        frame,
        f"Estado: {current_instruction(step)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
    )
    cv2.putText(
        frame,
        f"Capturadas: {captured_faces_text(captured_faces)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        2,
    )
    cv2.putText(
        frame,
        f"Estavel: {stable_count}/{stable_frames_required}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        2,
    )
    cv2.putText(
        frame,
        "[ESC] Sair",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        2,
    )


def current_instruction(step: CaptureStep) -> str:
    """Retorna a instrucao textual para a etapa atual."""

    if step.kind == "return_u":
        return "Volte a face TOPO (U) para a lente"

    if step.face is None:
        return "Prepare o cubo"

    face_name = FACE_NOMES.get(step.face, step.face)
    hint = CAPTURE_HINTS.get(step.face, "")
    return f"Mostre {face_name} {hint}".strip()


def captured_faces_text(captured_faces: dict[str, FaceGrid]) -> str:
    """Formata a lista de faces ja capturadas para o HUD."""

    return " ".join(face for face in FACE_ORDEM if face in captured_faces)


def show_windows(frame, result: ScanFrameResult) -> None:
    """Mostra as janelas de deteccao e de face recortada."""

    if result.annotated_face is not None:
        cv2.imshow(JANELA_FACE, result.annotated_face)
    else:
        destroy_window_if_open(JANELA_FACE)

    cv2.imshow(JANELA_DETECCAO, frame)


def destroy_window_if_open(window_name: str) -> None:
    """Fecha uma janela OpenCV sem propagar erro caso ela nao exista."""

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


if __name__ == "__main__":
    raise SystemExit(run())
