"""Modulo de acesso a camera usando OpenCV."""

from __future__ import annotations

import platform
import time
from types import TracebackType
from typing import ClassVar

import cv2
import numpy as np


class CameraConnectionError(RuntimeError):
    """Excecao lancada quando a camera nao pode ser aberta ou lida."""


class Camera:
    """Gerencia a conexao com a camera usada para capturar frames do cubo.

    A classe encapsula o objeto ``cv2.VideoCapture`` e garante que o recurso
    seja liberado ao final do uso por meio do protocolo de Context Manager.

    Args:
        camera_index: Indice opcional da camera. Quando informado, a classe usa
            este indice diretamente.
        preferred_name: Trecho opcional do nome da camera desejada. No Windows,
            quando o pygrabber estiver disponivel, a classe tenta priorizar uma
            camera cujo nome contenha esse texto.
        search_limit: Quantidade maxima de indices testados quando a camera
            precisa ser descoberta automaticamente.

    Raises:
        CameraConnectionError: Se a camera nao puder ser aberta.
    """

    FRAME_WIDTH: ClassVar[int] = 1920
    FRAME_HEIGHT: ClassVar[int] = 1080
    STARTUP_DELAY_SECONDS: ClassVar[float] = 1.5
    CAMERA_SEARCH_LIMIT: ClassVar[int] = 10

    def __init__(
        self,
        camera_index: int | None = None,
        preferred_name: str | None = None,
        search_limit: int | None = None,
    ) -> None:
        self._camera_index: int = (
            camera_index
            if camera_index is not None
            else self._find_camera_index(preferred_name, search_limit)
        )
        self._cap: cv2.VideoCapture = self._create_capture(self._camera_index)
        self._configure_capture()

    def __enter__(self) -> Camera:
        """Retorna a propria instancia para uso com ``with``."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Libera a camera ao sair do bloco de Context Manager."""

        self.release()

    def get_frame(self) -> np.ndarray:
        """Le um frame da camera.

        Returns:
            Frame capturado como um ``np.ndarray`` no formato retornado pelo
            OpenCV.

        Raises:
            CameraConnectionError: Se a leitura do frame falhar.
        """

        success, frame = self._cap.read()
        if not success or frame is None:
            raise CameraConnectionError("Erro: nao foi possivel ler frame da camera.")

        return frame

    def release(self) -> None:
        """Libera explicitamente o recurso da camera."""

        if self._cap.isOpened():
            self._cap.release()

    @classmethod
    def _find_camera_index(
        cls, preferred_name: str | None = None, search_limit: int | None = None
    ) -> int:
        """Descobre automaticamente a primeira camera disponivel."""

        limit = search_limit if search_limit is not None else cls.CAMERA_SEARCH_LIMIT
        named_cameras: dict[int, str] = {}
        if platform.system() == "Windows":
            named_cameras = cls._get_named_cameras_on_windows()

        candidates = cls._build_candidate_indexes(named_cameras, preferred_name, limit)
        first_available_index: int | None = None

        for index in candidates:
            is_available, supports_target_resolution = cls._probe_camera(index)
            if not is_available:
                continue

            if first_available_index is None:
                first_available_index = index

            if supports_target_resolution:
                return index

        if first_available_index is not None:
            return first_available_index

        raise CameraConnectionError("Erro: nenhuma camera disponivel foi encontrada.")

    @classmethod
    def _get_named_cameras_on_windows(cls) -> dict[int, str]:
        """Lista cameras com nomes usando pygrabber quando disponivel no Windows."""

        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            return {}

        try:
            graph = FilterGraph()
            device_names = [str(name) for name in graph.get_input_devices()]
        except Exception:
            return {}

        return dict(enumerate(device_names))

    @classmethod
    def _build_candidate_indexes(
        cls,
        named_cameras: dict[int, str],
        preferred_name: str | None,
        search_limit: int,
    ) -> list[int]:
        """Monta a ordem de indices que serao testados para abertura."""

        candidates: list[int] = []

        if preferred_name:
            preferred_name_upper = preferred_name.upper()
            for index, camera_name in named_cameras.items():
                if preferred_name_upper in camera_name.upper():
                    candidates.append(index)

        candidates.extend(named_cameras.keys())
        candidates.extend(range(search_limit))

        return list(dict.fromkeys(candidates))

    @classmethod
    def _probe_camera(cls, camera_index: int) -> tuple[bool, bool]:
        """Verifica se uma camera abre e se aceita a resolucao alvo."""

        temporary_capture = cls._open_capture(camera_index)
        try:
            if not temporary_capture.isOpened():
                return False, False

            temporary_capture.set(cv2.CAP_PROP_FRAME_WIDTH, cls.FRAME_WIDTH)
            temporary_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cls.FRAME_HEIGHT)

            current_width = int(temporary_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(temporary_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            supports_target_resolution = (
                current_width == cls.FRAME_WIDTH
                and current_height == cls.FRAME_HEIGHT
            )

            return True, supports_target_resolution
        finally:
            temporary_capture.release()

    @classmethod
    def _create_capture(cls, camera_index: int) -> cv2.VideoCapture:
        """Cria e valida o objeto ``cv2.VideoCapture``."""

        capture = cls._open_capture(camera_index)
        if not capture.isOpened():
            capture.release()
            raise CameraConnectionError(
                f"Erro: nao foi possivel abrir a camera no indice {camera_index}."
            )

        return capture

    @classmethod
    def _open_capture(cls, camera_index: int) -> cv2.VideoCapture:
        """Abre a camera usando DirectShow no Windows e backend padrao nos demais SOs."""

        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        return cv2.VideoCapture(camera_index, backend)

    def _configure_capture(self) -> None:
        """Configura resolucao e aguarda a inicializacao da camera."""

        time.sleep(self.STARTUP_DELAY_SECONDS)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)


__all__ = ["Camera", "CameraConnectionError"]
