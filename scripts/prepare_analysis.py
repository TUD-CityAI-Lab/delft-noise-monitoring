"""Build a read-only-analysis database from the prepared public data bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import duckdb
import pandas as pd

TABLES = ('loudness_per_hour', 'ir_per_hour',
          'intermittent_event_sums_per_sensor')
VIEWS = {'loudness': 'lake/scsm_loudness/date=*/sensor_id=*/*.parquet',
         'sources': 'lake/scsm_sources/date=*/sensor_id=*/*.parquet',
         'loudness_per_minute': 'marts/loudness_per_minute/date=*/sensor_id=*/*.parquet'}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def literal(value):
    return "'" + str(value).replace("'", "''") + "'"

def validate_bundle(data_root, verify_data=False):
    root = Path(data_root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported data manifest')
    records = manifest['files']
    names = [r['path'] for r in records]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest paths')
    required = {f'derived/{t}.parquet' for t in TABLES} | {
        'ref/sensors.tsv', 'ref/observation-timespans.csv', 'ref/delft-binnenstad-outline.csv'}
    if not required <= set(names):
        raise ValueError('Missing required derived/reference inputs')
    for record in records:
        path = (root / record['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Data manifest contains a path outside the bundle')
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing or changed input: {record["path"]}')
        if verify_data or record['path'].startswith(('derived/', 'ref/')):
            if sha256(path) != record['sha256']:
                raise ValueError(f'Checksum mismatch: {record["path"]}')
    for pattern in VIEWS.values():
        actual = {p.relative_to(root).as_posix() for p in root.glob(pattern)}
        expected = {p for p in names if Path(p).match(pattern)}
        if not actual or actual != expected:
            raise ValueError(f'Dataset file set differs from manifest: {pattern}')
    return manifest

def build_database(data_root, destination, verify_data=False):
    root, destination = Path(data_root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    manifest = validate_bundle(root, verify_data)
    records = {r['path']: r for r in manifest['files']}
    sensors = pd.read_csv(root/'ref/sensors.tsv', sep='\t', dtype=str).rename(columns={'ID':'sensor_id'})
    sensors['sensor_id'] = sensors.sensor_id.str.strip()
    for c in ['latitude', 'longitude']:
        sensors[c] = pd.to_numeric(sensors[c], errors='coerce')
    sensors = sensors[['sensor_id', 'latitude', 'longitude', 'height']].drop_duplicates()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='delft_setup_', dir=destination.parent) as folder:
        staged = Path(folder)/'analysis.duckdb'
        with duckdb.connect(str(staged)) as db:
            for name in TABLES:
                path = root/'derived'/f'{name}.parquet'
                db.execute(f'CREATE TABLE "{name}" AS SELECT * FROM read_parquet({literal(path.as_posix())})')
                if db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] != records[f'derived/{name}.parquet']['rows']:
                    raise ValueError(f'Row count mismatch: {name}')
            db.register('sensor_reference', sensors)
            db.execute("""CREATE TABLE sensors AS SELECT sensor_id::TEXT AS sensor_id,
                latitude::DOUBLE AS latitude, longitude::DOUBLE AS longitude,
                NULLIF(TRIM(height), '')::TEXT AS height FROM sensor_reference""")
            for name, pattern in VIEWS.items():
                db.execute(f'CREATE VIEW "{name}" AS SELECT * FROM read_parquet({literal((root/pattern).as_posix())}, hive_partitioning=true)')
        os.link(staged, destination)
    return {'database': str(destination), 'tables': len(TABLES)+1,
            'views': len(VIEWS), 'all_data_checksums_verified': verify_data}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--data-root', type=Path, default=Path('data'))
    build.add_argument('--destination', type=Path, required=True)
    build.add_argument('--verify-data', action='store_true', help='Check every lake/mart checksum as well as all derived/reference checksums')
    args = parser.parse_args()
    print(json.dumps(build_database(args.data_root, args.destination, args.verify_data), indent=2))

if __name__ == '__main__':
    main()
