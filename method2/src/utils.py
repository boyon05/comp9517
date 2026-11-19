"""Provides shared path and configuration utilities."""

from pathlib import Path
import yaml
MODULE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_ROOT.parent

def get_resolved_data_yaml(yaml_path=None):
    if yaml_path is None:
        yaml_path = MODULE_ROOT / 'configs' / 'data.yaml'
    else:
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path = (MODULE_ROOT / yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {yaml_path}')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    raw_path = data_cfg.get('path', '')
    target_path = Path(raw_path)
    if not target_path.is_absolute():
        candidate = (yaml_path.parent / raw_path).resolve()
        if not candidate.exists():
            candidate = (REPO_ROOT / 'common' / 'dataset').resolve()
        data_cfg['path'] = str(candidate)
    runtime_yaml = yaml_path.parent / '_runtime_data.yaml'
    with open(runtime_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)
    return str(runtime_yaml)
