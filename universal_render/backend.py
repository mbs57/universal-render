# universal_render/backend.py
from __future__ import annotations

import os
import sys
import platform
import warnings as _pywarnings
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

try:
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication
    QT_AVAILABLE = True
    QT_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    QGuiApplication = None
    QApplication = None
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = e


# ---------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------

_APP = None
_INITIALIZED = False


@dataclass
class RendererStatus:
    qt_available: bool = QT_AVAILABLE
    qt_import_error: Optional[str] = None
    initialized: bool = False
    app_created: bool = False
    app_class: Optional[str] = None
    platform_name: str = field(default_factory=lambda: platform.system())
    python_version: str = field(default_factory=lambda: platform.python_version())
    headless: bool = False
    offscreen_requested: bool = False
    offscreen_enabled: bool = False
    qt_qpa_platform: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_STATUS = RendererStatus(
    qt_import_error=str(QT_IMPORT_ERROR) if QT_IMPORT_ERROR else None
)


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------

def is_notebook_environment() -> bool:
    """
    Best-effort detection for Jupyter/Colab/Kaggle-like notebook environments.
    """
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        if shell is None:
            return False
        return True
    except Exception:
        return False


def is_colab_environment() -> bool:
    return "google.colab" in sys.modules


def is_kaggle_environment() -> bool:
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
        or "/kaggle/" in os.getcwd().replace("\\", "/")
    )


def is_headless_environment() -> bool:
    """
    Heuristic headless detection.

    On Linux, the absence of DISPLAY/WAYLAND_DISPLAY usually means headless.
    On Colab/Kaggle, assume headless.
    On Windows/macOS, default to False unless explicitly offscreen.
    """
    if is_colab_environment() or is_kaggle_environment():
        return True

    system_name = platform.system().lower()

    if system_name == "linux":
        display = os.environ.get("DISPLAY")
        wayland = os.environ.get("WAYLAND_DISPLAY")
        return not (display or wayland)

    return False


def _normalize_bool(value: Optional[bool], default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _ensure_offscreen_env(force: bool = False) -> bool:
    """
    Enable Qt offscreen mode by setting QT_QPA_PLATFORM=offscreen.

    Returns True if the environment variable is set to 'offscreen' after this call.
    """
    current = os.environ.get("QT_QPA_PLATFORM")

    if current:
        # Respect explicit user setting unless force=True
        if force:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            return True
        return current.lower() == "offscreen"

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    return True


def _clear_offscreen_on_native_platform() -> Optional[str]:
    """
    Drop QT_QPA_PLATFORM=offscreen on Windows and macOS.

    Qt's offscreen platform plugin carries no font engine on those systems,
    so every string rasterizes to nothing and figures come out blank with
    no error. Windows and macOS render correctly through their native
    plugin without a visible window, so offscreen is never needed there.
    Notebooks written for Linux commonly set the variable unconditionally,
    which is why this guard exists.

    Returns a warning message if the variable was cleared, else None.
    """
    system = platform.system().lower()
    if system not in ("windows", "darwin"):
        return None
    if (os.environ.get("QT_QPA_PLATFORM") or "").lower() != "offscreen":
        return None
    if is_colab_environment() or is_kaggle_environment():
        return None

    os.environ.pop("QT_QPA_PLATFORM", None)
    return (
        f"QT_QPA_PLATFORM=offscreen was cleared on {platform.system()}: the "
        "Qt offscreen plugin cannot rasterize text on this platform and "
        "would render every label blank. The native platform plugin is "
        "used instead and works without a visible window."
    )


def _warn_if_app_is_offscreen(app) -> Optional[str]:
    """
    Detect a Qt application that was already created under the offscreen
    platform on Windows or macOS.

    Qt applications are process-global and cannot change platform plugin
    after construction, so clearing the environment variable no longer
    helps at this point: every render will be blank. This happens when a
    notebook kernel imported Qt with QT_QPA_PLATFORM=offscreen earlier in
    the same session. The only remedy is restarting the process, so say
    so loudly rather than returning empty images.
    """
    system = platform.system().lower()
    if system not in ("windows", "darwin"):
        return None
    if is_colab_environment() or is_kaggle_environment():
        return None
    try:
        name = str(app.platformName()).lower()
    except Exception:
        return None
    if name != "offscreen":
        return None

    msg = (
        "Qt is already running with the 'offscreen' platform plugin on "
        f"{platform.system()}, which cannot rasterize text: every label "
        "will render blank. A Qt application cannot change platform after "
        "it starts, so please RESTART the kernel or process. Make sure "
        "QT_QPA_PLATFORM is not set to 'offscreen' before the first "
        "import (it is only needed on headless Linux)."
    )
    _pywarnings.warn(msg, RuntimeWarning, stacklevel=3)
    return msg


def _existing_app():
    """
    Return an existing Qt application instance if one exists.
    """
    if not QT_AVAILABLE:
        return None

    try:
        app = QApplication.instance()
        if app is not None:
            return app
    except Exception:
        pass

    try:
        app = QGuiApplication.instance()
        if app is not None:
            return app
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def init_renderer(
    headless: Optional[bool] = None,
    offscreen: bool = True,
    force: bool = False,
) -> Any:
    """
    Initialize a Qt application safely for multilingual text rendering.

    Parameters
    ----------
    headless:
        Whether to treat the current environment as headless.
        If None, a best-effort detection is used.
    offscreen:
        Whether to request Qt offscreen mode in headless environments.
    force:
        If True, re-run initialization logic even if already initialized.

    Returns
    -------
    app:
        The active QApplication/QGuiApplication instance.

    Notes
    -----
    - In Colab/Kaggle/headless Linux, offscreen mode is usually required.
    - This function tries to reuse an existing Qt app if one already exists.
    """
    global _APP, _INITIALIZED, _STATUS

    if not QT_AVAILABLE:
        raise RuntimeError(
            "PySide6 is not available. Install PySide6 to use universal_render.\n"
            f"Original import error: {QT_IMPORT_ERROR}"
        )

    inferred_headless = is_headless_environment()
    use_headless = _normalize_bool(headless, inferred_headless)

    warnings: List[str] = []

    # Must run before the QApplication is constructed: Qt reads the
    # platform plugin name from the environment at that moment.
    cleared = _clear_offscreen_on_native_platform()
    if cleared:
        warnings.append(cleared)

    if _INITIALIZED and not force:
        _STATUS.headless = use_headless
        _STATUS.offscreen_requested = bool(offscreen)
        _STATUS.qt_qpa_platform = os.environ.get("QT_QPA_PLATFORM")
        _STATUS.offscreen_enabled = (
            (_STATUS.qt_qpa_platform or "").lower() == "offscreen"
        )
        if warnings:
            _STATUS.warnings.extend(warnings)
        return _APP

    if use_headless and offscreen:
        enabled = _ensure_offscreen_env(force=force)
        if enabled:
            _STATUS.offscreen_enabled = True
        else:
            warnings.append(
                "Headless mode detected, but QT_QPA_PLATFORM is not set to 'offscreen'."
            )

    existing = _existing_app()
    if existing is not None:
        _APP = existing
        _INITIALIZED = True

        stale = _warn_if_app_is_offscreen(existing)
        if stale:
            warnings.append(stale)

        _STATUS.initialized = True
        _STATUS.app_created = True
        _STATUS.app_class = type(existing).__name__
        _STATUS.headless = use_headless
        _STATUS.offscreen_requested = bool(offscreen)
        _STATUS.qt_qpa_platform = os.environ.get("QT_QPA_PLATFORM")
        _STATUS.offscreen_enabled = (
            (_STATUS.qt_qpa_platform or "").lower() == "offscreen"
        )
        _STATUS.warnings = warnings
        return _APP

    argv = [sys.argv[0] if sys.argv else "universal_render"]

    try:
        _APP = QApplication(argv)
    except Exception as e:
        _STATUS.initialized = False
        _STATUS.app_created = False
        _STATUS.app_class = None
        _STATUS.headless = use_headless
        _STATUS.offscreen_requested = bool(offscreen)
        _STATUS.qt_qpa_platform = os.environ.get("QT_QPA_PLATFORM")
        _STATUS.offscreen_enabled = (
            (_STATUS.qt_qpa_platform or "").lower() == "offscreen"
        )
        _STATUS.warnings = warnings

        extra = []
        if use_headless and offscreen:
            extra.append(
                "This looks like a headless environment. Qt offscreen rendering was requested."
            )
        if is_colab_environment():
            extra.append(
                "Detected Google Colab. You may also need to install fonts and keep QT_QPA_PLATFORM=offscreen."
            )
        if is_kaggle_environment():
            extra.append(
                "Detected Kaggle. You may also need to install fonts and keep QT_QPA_PLATFORM=offscreen."
            )

        extra_text = "\n".join(extra)
        raise RuntimeError(
            "Failed to initialize Qt for universal_render.\n"
            f"Qt error: {e}\n"
            f"{extra_text}".strip()
        ) from e

    _INITIALIZED = True

    _STATUS.initialized = True
    _STATUS.app_created = True
    _STATUS.app_class = type(_APP).__name__
    _STATUS.headless = use_headless
    _STATUS.offscreen_requested = bool(offscreen)
    _STATUS.qt_qpa_platform = os.environ.get("QT_QPA_PLATFORM")
    _STATUS.offscreen_enabled = (
        (_STATUS.qt_qpa_platform or "").lower() == "offscreen"
    )
    _STATUS.warnings = warnings

    if use_headless and not _STATUS.offscreen_enabled:
        _STATUS.warnings.append(
            "Headless environment detected, but offscreen Qt platform is not active."
        )

    return _APP


def ensure_qt_application() -> Any:
    """
    Convenience wrapper for lazy initialization with safe defaults.
    """
    return init_renderer(headless=None, offscreen=True, force=False)


def get_renderer_status() -> Dict[str, Any]:
    """
    Return backend/Qt initialization status as a plain dictionary.
    """
    _STATUS.qt_available = QT_AVAILABLE
    _STATUS.qt_import_error = str(QT_IMPORT_ERROR) if QT_IMPORT_ERROR else None
    _STATUS.qt_qpa_platform = os.environ.get("QT_QPA_PLATFORM")
    _STATUS.offscreen_enabled = ((_STATUS.qt_qpa_platform or "").lower() == "offscreen")
    _STATUS.headless = is_headless_environment()
    return _STATUS.to_dict()


def check_environment() -> Dict[str, Any]:
    """
    Lightweight diagnostic snapshot of the current runtime.
    """
    warnings: List[str] = []

    qt_qpa = os.environ.get("QT_QPA_PLATFORM")
    headless = is_headless_environment()

    if not QT_AVAILABLE:
        warnings.append("PySide6 is not installed or failed to import.")

    if headless and (qt_qpa or "").lower() != "offscreen":
        warnings.append(
            "Headless environment detected but QT_QPA_PLATFORM is not 'offscreen'."
        )

    if is_colab_environment():
        warnings.append(
            "Running in Google Colab. Script fonts may need manual installation or registration."
        )

    if is_kaggle_environment():
        warnings.append(
            "Running in Kaggle. Script fonts may need manual installation or registration."
        )

    return {
        "qt_available": QT_AVAILABLE,
        "qt_import_error": str(QT_IMPORT_ERROR) if QT_IMPORT_ERROR else None,
        "headless": headless,
        "notebook": is_notebook_environment(),
        "colab": is_colab_environment(),
        "kaggle": is_kaggle_environment(),
        "qt_qpa_platform": qt_qpa,
        "warnings": warnings,
    }


def reset_renderer_state() -> None:
    """
    Reset only the Python-side tracking state.

    This does not destroy an already-running Qt application, because Qt apps are
    generally process-global and should not be forcefully torn down here.
    """
    global _APP, _INITIALIZED, _STATUS

    existing = _existing_app()
    _APP = existing
    _INITIALIZED = existing is not None

    _STATUS = RendererStatus(
        qt_import_error=str(QT_IMPORT_ERROR) if QT_IMPORT_ERROR else None
    )
    _STATUS.initialized = _INITIALIZED
    _STATUS.app_created = existing is not None
    _STATUS.app_class = type(existing).__name__ if existing is not None else None
    _STATUS.qt_qpa_platform = os.environ.get("QT_QPA_PLATFORM")
    _STATUS.offscreen_enabled = (
        ((_STATUS.qt_qpa_platform or "").lower() == "offscreen")
    )
    _STATUS.headless = is_headless_environment()