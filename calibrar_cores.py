"""Utilitario visual para calibrar os limites HSV das cores do cubo."""

from __future__ import annotations

from typing import Final, TypedDict

import cv2
import numpy as np

from src.utils.file_manager import FileAccessError, load_json, save_json
from src.vision.camera import Camera, CameraConnectionError

CALIBRATION_FILE: Final[str] = "cores_cubo.json"
TRACKBARS_WINDOW: Final[str] = "Trackbars"
WEBCAM_WINDOW: Final[str] = "Webcam Original"
MASK_WINDOW: Final[str] = "Mascara (Limiarizacao)"
ESC_KEY: Final[int] = 27

COLOR_KEYS: Final[dict[int, str]] = {
    ord("v"): "vermelho",
    ord("g"): "verde",
    ord("a"): "azul",
    ord("m"): "amarelo",
    ord("l"): "laranja",
    ord("b"): "branco",
}

STATUS_BY_COLOR: Final[dict[str, str]] = {
    "vermelho": "Vermelho atualizado!",
    "verde": "Verde atualizado!",
    "azul": "Azul atualizado!",
    "amarelo": "Amarelo atualizado!",
    "laranja": "Laranja atualizado!",
    "branco": "Branco atualizado!",
}


class ColorCalibration(TypedDict):
    """Limites HSV inferiores e superiores de uma cor."""

    lower: list[int]
    upper: list[int]


Calibrations = dict[str, ColorCalibration]


def empty(_value: int) -> None:
    """Callback vazio exigido pela API de trackbars do OpenCV."""


def run() -> int:
    """Executa a interface de calibracao HSV."""

    try:
        calibrations, status_message = load_existing_calibrations(CALIBRATION_FILE)
        create_trackbars()

        with Camera() as cam:
            while True:
                frame = cam.get_frame()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_color, upper_color, current_values = read_trackbar_values()
                mask = cv2.inRange(hsv, lower_color, upper_color)

                draw_detected_contours(frame, mask)
                draw_hud(frame, status_message)

                cv2.imshow(WEBCAM_WINDOW, frame)
                cv2.imshow(MASK_WINDOW, mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ESC_KEY:
                    break

                if key in COLOR_KEYS:
                    color_name = COLOR_KEYS[key]
                    calibrations[color_name] = current_values
                    status_message = STATUS_BY_COLOR[color_name]
                elif key == ord("s"):
                    save_json(CALIBRATION_FILE, calibrations)
                    status_message = "Arquivo 'cores_cubo.json' SALVO!"

        return 0
    except (CameraConnectionError, FileAccessError) as exc:
        print("\nNao foi possivel executar a calibracao.")
        print(str(exc))
        return 1
    finally:
        cv2.destroyAllWindows()


def load_existing_calibrations(filepath: str) -> tuple[Calibrations, str]:
    """Carrega calibracoes existentes ou inicia um dicionario vazio."""

    try:
        raw_calibrations = load_json(filepath)
    except FileAccessError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return {}, "Novo JSON. Ajuste as barras e grave a cor."
        raise

    return normalize_calibrations(raw_calibrations), (
        "JSON carregado! Ajuste e regrave a cor desejada."
    )


def normalize_calibrations(raw_calibrations: dict[str, object]) -> Calibrations:
    """Converte o dicionario JSON para o tipo esperado pelo calibrador."""

    calibrations: Calibrations = {}
    for color_name, raw_value in raw_calibrations.items():
        if not isinstance(raw_value, dict):
            continue

        lower = raw_value.get("lower")
        upper = raw_value.get("upper")
        if not isinstance(lower, list) or not isinstance(upper, list):
            continue

        calibrations[color_name] = {
            "lower": [int(value) for value in lower],
            "upper": [int(value) for value in upper],
        }

    return calibrations


def create_trackbars() -> None:
    """Cria a janela e as trackbars de calibracao HSV."""

    cv2.namedWindow(TRACKBARS_WINDOW)
    cv2.resizeWindow(TRACKBARS_WINDOW, 640, 250)
    cv2.createTrackbar("Hue Min", TRACKBARS_WINDOW, 0, 179, empty)
    cv2.createTrackbar("Hue Max", TRACKBARS_WINDOW, 179, 179, empty)
    cv2.createTrackbar("Sat Min", TRACKBARS_WINDOW, 0, 255, empty)
    cv2.createTrackbar("Sat Max", TRACKBARS_WINDOW, 255, 255, empty)
    cv2.createTrackbar("Val Min", TRACKBARS_WINDOW, 0, 255, empty)
    cv2.createTrackbar("Val Max", TRACKBARS_WINDOW, 255, 255, empty)


def read_trackbar_values() -> tuple[np.ndarray, np.ndarray, ColorCalibration]:
    """Le os valores atuais das trackbars HSV."""

    h_min = cv2.getTrackbarPos("Hue Min", TRACKBARS_WINDOW)
    h_max = cv2.getTrackbarPos("Hue Max", TRACKBARS_WINDOW)
    s_min = cv2.getTrackbarPos("Sat Min", TRACKBARS_WINDOW)
    s_max = cv2.getTrackbarPos("Sat Max", TRACKBARS_WINDOW)
    v_min = cv2.getTrackbarPos("Val Min", TRACKBARS_WINDOW)
    v_max = cv2.getTrackbarPos("Val Max", TRACKBARS_WINDOW)

    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max])
    current_values: ColorCalibration = {
        "lower": [int(h_min), int(s_min), int(v_min)],
        "upper": [int(h_max), int(s_max), int(v_max)],
    }

    return lower_color, upper_color, current_values


def draw_detected_contours(frame: np.ndarray, mask: np.ndarray) -> None:
    """Desenha contornos grandes encontrados pela mascara ativa."""

    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)


def draw_hud(frame: np.ndarray, status_message: str) -> None:
    """Desenha as instrucoes e o status no frame da webcam."""

    cv2.putText(
        frame,
        "Gravar Cor: [V]ermelho [G]Verde [A]zul [M]Amarelo [L]Laranja [B]Branco",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Salvar JSON: Aperte [S] | Sair: Aperte [ESC]",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Status: {status_message}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


if __name__ == "__main__":
    raise SystemExit(run())
