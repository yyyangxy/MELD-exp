from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or PROJECT_ROOT).joinpath(path).resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_experiment_output_dir(config: dict[str, Any]) -> Path:
    train_cfg = config.get("train", {})
    output_root = resolve_path(train_cfg.get("output_dir", "outputs"), PROJECT_ROOT)
    run_cfg = config.get("run", {})
    if not bool(run_cfg.get("enabled", False)):
        return output_root

    group = _safe_path_component(str(run_cfg.get("group", "default")))
    name = _safe_path_component(str(run_cfg.get("name", "run")))
    runs_root = ensure_dir(output_root / "runs" / group)

    explicit_dir = run_cfg.get("dir")
    if explicit_dir:
        run_dir = resolve_path(explicit_dir, PROJECT_ROOT)
    else:
        run_id = run_cfg.get("id")
        if run_id:
            run_dir = runs_root / _safe_path_component(str(run_id))
        elif bool(run_cfg.get("reuse_existing", True)):
            matches = sorted(runs_root.glob(f"*_{name}"))
            run_dir = matches[-1] if matches else runs_root / f"{_timestamp()}_{name}"
        else:
            run_dir = runs_root / f"{_timestamp()}_{name}"

    ensure_dir(run_dir)
    _write_run_metadata(run_dir, config)
    return run_dir


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _parse_minimal_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    data["_config_path"] = str(path)
    data["_config_dir"] = str(path.parent)
    return data


def _write_run_metadata(run_dir: Path, config: dict[str, Any]) -> None:
    config_path = run_dir / "config.yaml"
    latest_config_path = run_dir / "config_latest.yaml"
    public_config = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    rendered_config = _render_config(public_config)
    if not config_path.exists():
        config_path.write_text(rendered_config, encoding="utf-8")
    latest_config_path.write_text(rendered_config, encoding="utf-8")

    notes_path = run_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Experiment Notes\n\n"
            "## Purpose\n\n"
            "- \n\n"
            "## Observations\n\n"
            "- \n",
            encoding="utf-8",
        )


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _render_config(config: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    except ModuleNotFoundError:
        return repr(config)


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Small YAML subset parser for the repo configs when PyYAML is absent."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(item.strip()) for item in body.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def resolve_data_root(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", {})
    mode = data_cfg.get("data_root_mode", "auto")

    server_root = resolve_path(data_cfg.get("server_data_root", ""), PROJECT_ROOT)
    local_root = resolve_path(data_cfg.get("local_data_root", ""), PROJECT_ROOT)

    if mode == "server":
        return server_root
    if mode == "local":
        return local_root
    if server_root.exists():
        return server_root
    if local_root.exists():
        return local_root
    raise FileNotFoundError(
        "No MELD data root found. Set data.data_root_mode and root paths in config."
    )
