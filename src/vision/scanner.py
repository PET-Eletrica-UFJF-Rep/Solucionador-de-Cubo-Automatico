"""Scanner de faces do cubo baseado em frames OpenCV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import cv2
import numpy as np

from src.utils.helpers import ordenar_pontos

ColorName: TypeAlias = str
FaceGrid: TypeAlias = list[list[ColorName]]
HSVBounds: TypeAlias = Mapping[str, Sequence[int]]
RawCalibrations: TypeAlias = dict[ColorName, HSVBounds]
ColorMaskMap: TypeAlias = dict[ColorName, np.ndarray]
ColorRangeMap: TypeAlias = dict[ColorName, tuple[np.ndarray, np.ndarray]]


class ScannerCalibrationError(ValueError):
    """Excecao lancada quando as calibracoes HSV nao estao no formato esperado."""


@dataclass(slots=True)
class ScanFrameResult:
    """Resultado de uma tentativa de leitura de face."""

    grid: FaceGrid | None
    center_color: ColorName
    contour: np.ndarray | None
    warped_face: np.ndarray | None
    annotated_frame: np.ndarray | None = None
    annotated_face: np.ndarray | None = None

    @property
    def found_cube(self) -> bool:
        """Indica se houve contorno travado para leitura."""

        return self.contour is not None

    @property
    def has_valid_grid(self) -> bool:
        """Indica se todos os quadrados da grade receberam uma cor."""

        if self.grid is None:
            return False

        return all(
            color != FaceScanner.UNDEFINED_COLOR
            for row in self.grid
            for color in row
        )


class FaceScanner:
    """Processa frames da camera e detecta as cores de uma face 3x3.

    A classe recebe calibracoes HSV prontas por injecao de dependencia. Ela nao
    le arquivos, nao instancia camera e nao contem loop principal de captura.
    """

    UNDEFINED_COLOR: ClassVar[str] = "Nenhum"

    WARPED_SIZE: ClassVar[int] = 300
    GRID_SIZE: ClassVar[int] = 3
    SQUARE_SIZE: ClassVar[int] = WARPED_SIZE // GRID_SIZE
    READ_MARGIN: ClassVar[int] = 30
    READ_THRESHOLD: ClassVar[float] = 0.15

    ROI_SIZE: ClassVar[int] = 300
    CANNY_LOW_THRESHOLD: ClassVar[int] = 30
    CANNY_HIGH_THRESHOLD: ClassVar[int] = 80
    GAUSSIAN_KERNEL: ClassVar[tuple[int, int]] = (7, 7)
    MORPH_KERNEL_SIZE: ClassVar[tuple[int, int]] = (7, 7)
    MORPH_CLOSE_ITERATIONS: ClassVar[int] = 3
    MIN_CONTOUR_AREA: ClassVar[float] = 15000.0
    APPROX_POLY_FACTOR: ClassVar[float] = 0.06
    MIN_ASPECT_RATIO: ClassVar[float] = 0.6
    MAX_ASPECT_RATIO: ClassVar[float] = 1.4

    CURRENT_POINTS_WEIGHT: ClassVar[float] = 0.7
    PREVIOUS_POINTS_WEIGHT: ClassVar[float] = 0.3
    MAX_LOST_FRAMES: ClassVar[int] = 10

    ROI_GUIDE_COLOR: ClassVar[tuple[int, int, int]] = (255, 255, 0)
    CONTOUR_COLOR: ClassVar[tuple[int, int, int]] = (0, 255, 0)
    GRID_LINE_COLOR: ClassVar[tuple[int, int, int]] = (100, 100, 100)
    DEFAULT_DRAW_COLOR: ClassVar[tuple[int, int, int]] = (255, 255, 255)
    VISUAL_COLORS: ClassVar[dict[ColorName, tuple[int, int, int]]] = {
        "vermelho": (0, 0, 255),
        "verde": (0, 255, 0),
        "azul": (255, 0, 0),
        "amarelo": (0, 255, 255),
        "laranja": (0, 165, 255),
        "branco": (255, 255, 255),
    }

    def __init__(self, calibrations: RawCalibrations) -> None:
        self._color_ranges: ColorRangeMap = self._normalize_calibrations(calibrations)
        self._smoothed_points: np.ndarray | None = None
        self._lost_frames: int = 0

    def reset_tracking(self) -> None:
        """Limpa a suavizacao temporal do contorno."""

        self._smoothed_points = None
        self._lost_frames = 0

    def scan_frame(
        self, frame: np.ndarray, *, draw_annotations: bool = True
    ) -> ScanFrameResult:
        """Processa um frame e tenta ler a grade 3x3 da face visivel.

        Args:
            frame: Frame BGR vindo da camera.
            draw_annotations: Quando verdadeiro, devolve copias anotadas do frame
                original e da face recortada.

        Returns:
            Um ``ScanFrameResult`` com a grade lida quando a face for encontrada.
        """

        annotated_frame = frame.copy() if draw_annotations else None
        if annotated_frame is not None:
            self._draw_roi_guide(annotated_frame)

        raw_contour = self._find_cube_contour(frame)
        tracked_contour = self._update_tracked_contour(raw_contour)

        if tracked_contour is None:
            return ScanFrameResult(
                grid=None,
                center_color=self.UNDEFINED_COLOR,
                contour=None,
                warped_face=None,
                annotated_frame=annotated_frame,
                annotated_face=None,
            )

        if annotated_frame is not None:
            self._draw_contour(annotated_frame, tracked_contour)

        warped_face = self._warp_perspective(frame, tracked_contour)
        grid, center_color = self._read_grid_colors(warped_face)
        annotated_face = (
            self._draw_grid_annotations(warped_face, grid) if draw_annotations else None
        )

        return ScanFrameResult(
            grid=grid,
            center_color=center_color,
            contour=tracked_contour.copy(),
            warped_face=warped_face,
            annotated_frame=annotated_frame,
            annotated_face=annotated_face,
        )

    def _find_cube_contour(self, frame: np.ndarray) -> np.ndarray | None:
        """Encontra o contorno quadrilateral do cubo no ROI central do frame."""

        x_roi, y_roi, frame_roi = self._extract_center_roi(frame)

        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.GAUSSIAN_KERNEL, 0)

        edges = cv2.Canny(
            blurred, self.CANNY_LOW_THRESHOLD, self.CANNY_HIGH_THRESHOLD
        )
        kernel = np.ones(self.MORPH_KERNEL_SIZE, np.uint8)
        closed = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, kernel, iterations=self.MORPH_CLOSE_ITERATIONS
        )

        contours, _hierarchy = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area <= self._scaled_min_contour_area(frame_roi):
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.APPROX_POLY_FACTOR * perimeter, True)

            if len(approx) != 4:
                continue

            _x, _y, width, height = cv2.boundingRect(approx)
            aspect_ratio = float(width) / height
            if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                offset = np.array([[[x_roi, y_roi]]], dtype=approx.dtype)
                return approx + offset

        return None

    def _warp_perspective(self, frame: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Extrai a face do cubo como imagem plana 300x300."""

        ordered_points = ordenar_pontos(contour)
        destination_points = np.float32(
            [
                [0, 0],
                [self.WARPED_SIZE, 0],
                [self.WARPED_SIZE, self.WARPED_SIZE],
                [0, self.WARPED_SIZE],
            ]
        )
        perspective_matrix = cv2.getPerspectiveTransform(
            np.float32(ordered_points), destination_points
        )

        return cv2.warpPerspective(
            frame, perspective_matrix, (self.WARPED_SIZE, self.WARPED_SIZE)
        )

    def _read_grid_colors(self, warped_face: np.ndarray) -> tuple[FaceGrid, ColorName]:
        """Le as cores dos nove quadrados da face usando mascaras HSV."""

        hsv_face = cv2.cvtColor(warped_face, cv2.COLOR_BGR2HSV)
        masks = self._build_color_masks(hsv_face)
        grid: FaceGrid = []

        for row_index in range(self.GRID_SIZE):
            row_colors: list[ColorName] = []

            for column_index in range(self.GRID_SIZE):
                x = column_index * self.SQUARE_SIZE
                y = row_index * self.SQUARE_SIZE
                color_name, pixel_count = self._choose_square_color(masks, x, y)

                if (
                    self._is_detection_confident(pixel_count)
                    and color_name != self.UNDEFINED_COLOR
                ):
                    row_colors.append(color_name)
                else:
                    row_colors.append(self.UNDEFINED_COLOR)

            grid.append(row_colors)

        return grid, grid[1][1]

    def _extract_center_roi(self, frame: np.ndarray) -> tuple[int, int, np.ndarray]:
        """Recorta o ROI central onde o usuario alinha o cubo."""

        frame_height, frame_width = frame.shape[:2]
        roi_size = min(self.ROI_SIZE, frame_width, frame_height)
        x_roi = (frame_width - roi_size) // 2
        y_roi = (frame_height - roi_size) // 2

        return x_roi, y_roi, frame[y_roi : y_roi + roi_size, x_roi : x_roi + roi_size]

    def _scaled_min_contour_area(self, frame_roi: np.ndarray) -> float:
        """Ajusta a area minima caso o frame seja menor que o ROI padrao."""

        roi_height, roi_width = frame_roi.shape[:2]
        scale = (roi_width * roi_height) / float(self.ROI_SIZE * self.ROI_SIZE)
        return self.MIN_CONTOUR_AREA * scale

    def _update_tracked_contour(self, contour: np.ndarray | None) -> np.ndarray | None:
        """Atualiza a suavizacao temporal do contorno detectado."""

        if contour is not None:
            current_points = ordenar_pontos(contour)
            if self._smoothed_points is None:
                self._smoothed_points = current_points
            else:
                self._smoothed_points = (
                    current_points * self.CURRENT_POINTS_WEIGHT
                    + self._smoothed_points * self.PREVIOUS_POINTS_WEIGHT
                )
            self._lost_frames = 0
        else:
            self._lost_frames += 1
            if self._lost_frames > self.MAX_LOST_FRAMES:
                self._smoothed_points = None

        if self._smoothed_points is None:
            return None

        return self._smoothed_points.copy()

    def _build_color_masks(self, hsv_face: np.ndarray) -> ColorMaskMap:
        """Monta as mascaras binarias para cada cor calibrada."""

        return {
            color_name: cv2.inRange(hsv_face, lower_bound, upper_bound)
            for color_name, (lower_bound, upper_bound) in self._color_ranges.items()
        }

    def _choose_square_color(
        self, masks: ColorMaskMap, x: int, y: int
    ) -> tuple[ColorName, int]:
        """Escolhe a cor dominante dentro da area segura de um quadrado."""

        best_color = self.UNDEFINED_COLOR
        best_pixel_count = 0

        for color_name, mask in masks.items():
            square_roi = mask[
                y + self.READ_MARGIN : y + self.SQUARE_SIZE - self.READ_MARGIN,
                x + self.READ_MARGIN : x + self.SQUARE_SIZE - self.READ_MARGIN,
            ]
            pixel_count = cv2.countNonZero(square_roi)

            if pixel_count > best_pixel_count:
                best_pixel_count = pixel_count
                best_color = color_name

        return best_color, best_pixel_count

    def _is_detection_confident(self, pixel_count: int) -> bool:
        """Verifica se a contagem de pixels passa o limiar da area interna."""

        inner_area = (self.SQUARE_SIZE - 2 * self.READ_MARGIN) ** 2
        return pixel_count > inner_area * self.READ_THRESHOLD

    def _draw_roi_guide(self, frame: np.ndarray) -> None:
        """Desenha a guia central de alinhamento no frame anotado."""

        x_roi, y_roi, frame_roi = self._extract_center_roi(frame)
        roi_height, roi_width = frame_roi.shape[:2]
        cv2.rectangle(
            frame,
            (x_roi, y_roi),
            (x_roi + roi_width, y_roi + roi_height),
            self.ROI_GUIDE_COLOR,
            2,
        )
        cv2.putText(
            frame,
            "Alinhe o cubo aqui",
            (x_roi + 10, y_roi + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.ROI_GUIDE_COLOR,
            2,
        )

    def _draw_contour(self, frame: np.ndarray, contour: np.ndarray) -> None:
        """Desenha o contorno suavizado no frame anotado."""

        contour_points = np.int32(contour).reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [contour_points],
            isClosed=True,
            color=self.CONTOUR_COLOR,
            thickness=3,
        )

    def _draw_grid_annotations(
        self, warped_face: np.ndarray, grid: FaceGrid
    ) -> np.ndarray:
        """Desenha as leituras de cor sobre a face recortada."""

        face_visualization = warped_face.copy()

        for row_index, row in enumerate(grid):
            for column_index, color_name in enumerate(row):
                x = column_index * self.SQUARE_SIZE
                y = row_index * self.SQUARE_SIZE
                detected = color_name != self.UNDEFINED_COLOR
                draw_color = (
                    self.VISUAL_COLORS.get(color_name, self.DEFAULT_DRAW_COLOR)
                    if detected
                    else self.DEFAULT_DRAW_COLOR
                )

                cv2.rectangle(
                    face_visualization,
                    (x, y),
                    (x + self.SQUARE_SIZE, y + self.SQUARE_SIZE),
                    self.GRID_LINE_COLOR,
                    1,
                )
                cv2.rectangle(
                    face_visualization,
                    (x + self.READ_MARGIN, y + self.READ_MARGIN),
                    (
                        x + self.SQUARE_SIZE - self.READ_MARGIN,
                        y + self.SQUARE_SIZE - self.READ_MARGIN,
                    ),
                    draw_color,
                    2,
                )

                if detected:
                    cv2.putText(
                        face_visualization,
                        color_name[:3].upper(),
                        (x + 30, y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        draw_color,
                        2,
                    )

        center_color = grid[1][1]
        center_draw_color = self.VISUAL_COLORS.get(
            center_color, self.DEFAULT_DRAW_COLOR
        )
        cv2.drawMarker(
            face_visualization,
            (self.WARPED_SIZE // 2, self.WARPED_SIZE // 2),
            center_draw_color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

        if center_color != self.UNDEFINED_COLOR:
            cv2.putText(
                face_visualization,
                f"CENTRO: {center_color.upper()}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                center_draw_color,
                2,
            )

        return face_visualization

    @classmethod
    def _normalize_calibrations(cls, calibrations: RawCalibrations) -> ColorRangeMap:
        """Copia e normaliza as calibracoes para arrays ``uint8`` do OpenCV."""

        if not calibrations:
            raise ScannerCalibrationError("Erro: calibracoes HSV vazias.")

        normalized: ColorRangeMap = {}
        for color_name, limits in calibrations.items():
            if "lower" not in limits or "upper" not in limits:
                raise ScannerCalibrationError(
                    f"Erro: calibracao da cor {color_name!r} deve conter lower e upper."
                )

            lower_bound = cls._build_hsv_bound(color_name, "lower", limits["lower"])
            upper_bound = cls._build_hsv_bound(color_name, "upper", limits["upper"])
            normalized[color_name] = (lower_bound, upper_bound)

        return normalized

    @staticmethod
    def _build_hsv_bound(
        color_name: ColorName, field_name: str, raw_bound: Sequence[int]
    ) -> np.ndarray:
        """Valida um limite HSV e o converte para array ``uint8``."""

        bound = np.asarray(raw_bound)
        if bound.shape != (3,):
            raise ScannerCalibrationError(
                f"Erro: {field_name} da cor {color_name!r} deve ter 3 valores HSV."
            )

        if np.any(bound < 0) or np.any(bound > 255):
            raise ScannerCalibrationError(
                f"Erro: {field_name} da cor {color_name!r} deve estar entre 0 e 255."
            )

        return bound.astype(np.uint8)


__all__ = ["FaceScanner", "ScanFrameResult", "ScannerCalibrationError"]
