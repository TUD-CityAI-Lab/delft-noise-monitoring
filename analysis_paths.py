"""Paths for analysis of prepared Delft data (no ingestion or database writes)."""
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent


def data_root():
    return Path(os.environ.get('DELFT_DATA_ROOT', ROOT / 'data')).resolve()


def database_path():
    return Path(os.environ.get('DELFT_DATABASE', ROOT / 'analysis.duckdb')).resolve()


def output_path(relative):
    """Resolve a generated output path under DELFT_OUTPUT_ROOT."""
    return Path(os.environ.get('DELFT_OUTPUT_ROOT', ROOT)).resolve() / relative


def connect_analysis():
    path = database_path()
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run python -m scripts.prepare_analysis build --help')
    config = {}
    if 'DELFT_THREADS' in os.environ:
        threads = int(os.environ['DELFT_THREADS'])
        if threads < 1:
            raise ValueError('DELFT_THREADS must be positive')
        config['threads'] = threads
    if 'DELFT_MEMORY_LIMIT' in os.environ:
        config['memory_limit'] = os.environ['DELFT_MEMORY_LIMIT']
    return duckdb.connect(str(path), read_only=True, config=config)


def require_file(path, purpose):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{purpose} requires {path}. See README.md for input requirements.')
    return path

def survey_input(filename):
    """Private survey files are supplied separately, never bundled with public data."""
    root = Path(os.environ.get('DELFT_SURVEY_ROOT', data_root() / 'survey'))
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'Survey analysis requires {filename}. Survey responses and linkage are not public; '
            'request access from the authors and set DELFT_SURVEY_ROOT to their containing directory.')
    return path

def bundle_fingerprint():
    import hashlib
    return hashlib.sha256((data_root() / 'manifest.json').read_bytes()).hexdigest()
