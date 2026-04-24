import yaml
import numpy as np
import copy
import torch
import os
import logging
from typing import Iterable, Optional
from matplotlib import cm


_LOG_ROOT_NAME = "omnimap"
_LOGGING_CONFIGURED = False
LOG_SECTIONS = ("main", "tsdf", "gaussian", "fisher", "planner", "profile")
_SECTION_SET = set(LOG_SECTIONS)
_SECTION_ALIASES = {
    "main": "main",
    "entry": "main",
    "entrypoint": "main",
    "tsdf": "tsdf",
    "gs": "gaussian",
    "gaussian": "gaussian",
    "fisher": "fisher",
    "planner": "planner",
    "policy": "planner",
    "nbv": "planner",
    "profile": "profile",
    "timeit": "profile",
}
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
# Black-background friendly bright palette (avoid deep/dim colors).
_SECTION_COLOR_MAP = {
    "main": "\033[38;5;117m",      # light cyan
    "tsdf": "\033[38;5;156m",      # light green
    "gaussian": "\033[38;5;223m",  # warm light orange
    "fisher": "\033[38;5;219m",    # light pink
    "planner": "\033[38;5;153m",   # light blue
    "profile": "\033[38;5;222m",   # light amber
}
_LEVEL_COLOR_MAP = {
    "DEBUG": "\033[38;5;151m",     # mint
    "INFO": "\033[38;5;111m",      # bright sky
    "WARNING": "\033[38;5;221m",   # bright yellow
    "ERROR": "\033[38;5;203m",     # soft red
    "CRITICAL": "\033[38;5;199m",  # magenta-red
}
_MESSAGE_COLOR_BY_LEVEL = {
    "DEBUG": "\033[38;5;188m",
    "INFO": "\033[38;5;255m",
    "WARNING": "\033[38;5;229m",
    "ERROR": "\033[38;5;224m",
    "CRITICAL": "\033[38;5;225m",
}


def _normalize_section(section: str) -> str:
    key = str(section).strip().lower()
    if key == "all":
        return "all"
    mapped = _SECTION_ALIASES.get(key, key)
    if mapped not in _SECTION_SET:
        raise ValueError(
            f"Unknown log section: {section!r}. Expected one of {sorted(_SECTION_SET)} or 'all'."
        )
    return mapped


def _resolve_sections(sections: Optional[Iterable[str]]) -> set[str]:
    if not sections:
        return set(_SECTION_SET)
    normalized = {_normalize_section(item) for item in sections}
    if "all" in normalized:
        return set(_SECTION_SET)
    return normalized


def _infer_section_from_logger_name(logger_name: str) -> str:
    parts = str(logger_name).split(".")
    for part in reversed(parts):
        key = str(part).strip().lower()
        if key in _SECTION_ALIASES:
            mapped = _SECTION_ALIASES[key]
            if mapped in _SECTION_SET:
                return mapped
    return "main"


class SectionFilter(logging.Filter):
    def __init__(self, enabled_sections: set[str], min_level: int):
        super().__init__()
        self.enabled_sections = set(enabled_sections)
        self.min_level = int(min_level)

    def filter(self, record: logging.LogRecord) -> bool:
        section = getattr(record, "section", None)
        if section is None:
            section = _infer_section_from_logger_name(record.name)
            setattr(record, "section", section)
        if section not in self.enabled_sections:
            return False
        return int(record.levelno) >= self.min_level


class SectionFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "section"):
            setattr(record, "section", _infer_section_from_logger_name(record.name))
        return super().format(record)


class ColorSectionFormatter(SectionFormatter):
    def __init__(self, *args, enable_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_color = bool(enable_color)

    def format(self, record: logging.LogRecord) -> str:
        if not self.enable_color:
            return super().format(record)
        if not hasattr(record, "section"):
            setattr(record, "section", _infer_section_from_logger_name(record.name))

        level_name = str(getattr(record, "levelname", "INFO")).upper()
        section_name = str(getattr(record, "section", "main")).lower()
        level_color = _LEVEL_COLOR_MAP.get(level_name, "\033[38;5;252m")
        section_color = _SECTION_COLOR_MAP.get(section_name, "\033[38;5;252m")
        message_color = _MESSAGE_COLOR_BY_LEVEL.get(level_name, "\033[38;5;252m")

        # Save originals and colorize selected fields.
        orig_levelname = record.levelname
        orig_section = getattr(record, "section", section_name)
        orig_message = record.msg
        try:
            record.levelname = (
                f"{_ANSI_BOLD}{level_color}{orig_levelname}{_ANSI_RESET}"
            )
            setattr(
                record,
                "section",
                f"{_ANSI_BOLD}{section_color}{orig_section}{_ANSI_RESET}",
            )
            if isinstance(orig_message, str):
                record.msg = f"{message_color}{orig_message}{_ANSI_RESET}"
            return super().format(record)
        finally:
            record.levelname = orig_levelname
            setattr(record, "section", orig_section)
            record.msg = orig_message


class SectionLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("section", self.extra["section"])
        return msg, kwargs


def _coerce_level(level):
    if level is None:
        return None
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        value = getattr(logging, level.upper(), None)
        if isinstance(value, int):
            return value
    raise ValueError(f"Unsupported logging level: {level!r}")


def configure_logging(
    *,
    profile: str = "default",
    level=None,
    log_file: Optional[str] = None,
    enabled_sections: Optional[Iterable[str]] = None,
    min_console_level=None,
    force: bool = False,
):
    """Configure a shared logger used by all entrypoints and backend modules."""
    global _LOGGING_CONFIGURED

    logger = logging.getLogger(_LOG_ROOT_NAME)
    if _LOGGING_CONFIGURED and not force:
        return logger

    profile = str(profile).lower()
    if profile not in {"quiet", "default", "debug"}:
        raise ValueError(f"Unknown log profile: {profile!r}")

    console_level_map = {
        "quiet": logging.WARNING,
        "default": logging.INFO,
        "debug": logging.DEBUG,
    }
    file_level_map = {
        "quiet": logging.INFO,
        "default": logging.DEBUG,
        "debug": logging.DEBUG,
    }

    override_level = _coerce_level(level)
    console_level = override_level if override_level is not None else console_level_map[profile]
    file_level = override_level if override_level is not None else file_level_map[profile]
    min_level = _coerce_level(min_console_level)
    if min_level is None:
        min_level = console_level
    sections = _resolve_sections(enabled_sections)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    plain_formatter = SectionFormatter(
        "%(asctime)s | %(levelname)s | %(section)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_stream = getattr(logging.StreamHandler(), "stream", None)
    use_color = bool(getattr(console_stream, "isatty", lambda: False)())
    color_formatter = ColorSectionFormatter(
        "%(asctime)s | %(levelname)s | %(section)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        enable_color=use_color,
    )
    section_filter = SectionFilter(enabled_sections=sections, min_level=min_level)

    console_handler = logging.StreamHandler()
    # Handler level must not be stricter than the section filter threshold,
    # otherwise DEBUG lines may be dropped before SectionFilter sees them.
    console_emit_level = min(int(console_level), int(min_level))
    console_handler.setLevel(console_emit_level)
    console_handler.setFormatter(color_formatter)
    console_handler.addFilter(section_filter)
    logger.addHandler(console_handler)

    if log_file:
        log_file_path = os.fspath(log_file)
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    _LOGGING_CONFIGURED = True
    logger.debug(
        "日志系统已初始化：profile=%s console=%s emit=%s file=%s sections=%s min_console_level=%s path=%s",
        profile,
        logging.getLevelName(console_level),
        logging.getLevelName(console_emit_level),
        logging.getLevelName(file_level),
        ",".join(sorted(sections)),
        logging.getLevelName(min_level),
        log_file if log_file else "disabled",
    )
    return logger


def get_logger(name: Optional[str] = None):
    if not name:
        return logging.getLogger(_LOG_ROOT_NAME)
    return logging.getLogger(f"{_LOG_ROOT_NAME}.{name}")


def get_section_logger(name: Optional[str], section: str):
    resolved = _normalize_section(section)
    if resolved == "all":
        resolved = "main"
    return SectionLoggerAdapter(get_logger(name), {"section": resolved})


def should_log_step(step_idx: int, log_every: int) -> bool:
    try:
        n = int(log_every)
    except Exception:
        n = 1
    if n <= 1:
        return True
    return int(step_idx) % n == 0


def Log(*args, tag="GaussianSplatting"):
    """Legacy logging shim kept for existing modules."""
    message = " ".join(str(arg) for arg in args)
    logger_name = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
    inferred_section = _infer_section_from_logger_name(logger_name)
    get_section_logger(f"legacy.{logger_name}", inferred_section).info(message)


def _resolve_paths(cfg, config_dir):
    """
    Resolve relative paths in the 'path' section of config.
    Paths starting with './' or '../' are resolved relative to config_dir.
    Absolute paths (starting with '/') are left unchanged.
    
    Args:
        cfg (dict): config dictionary.
        config_dir (str): directory where the config file is located.
    """
    if 'path' not in cfg:
        return
    
    for key, value in cfg['path'].items():
        if isinstance(value, str) and not os.path.isabs(value):
            # Relative path - resolve it relative to config directory
            cfg['path'][key] = os.path.normpath(os.path.join(config_dir, value))


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # Get the directory where config file is located (for resolving relative paths)
    config_dir = os.path.dirname(os.path.abspath(path))
    
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)
    
    # Resolve relative paths in the 'path' section
    _resolve_paths(cfg, config_dir)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def colorize_np(x, cmap_name='jet', range=None):
    if range is not None:
        vmin, vmax = range
    else:
        vmin, vmax = np.percentile(x, (1, 99))

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]
    return x_new


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
