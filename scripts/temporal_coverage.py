"""Read-only deployment coverage summary; run from the repository root in delft-sensors.

Writes only out/temporal_coverage. Raw SPL dates/counts come from Parquet
row-group statistics, checked against partition dates. Scans only the required
SED and minute-mart columns, never reruns preprocessing or changes the database.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from analysis_paths import connect_analysis, data_root, output_path, bundle_fingerprint
OUT = output_path("out/temporal_coverage")
CLASSES = "vehicle honking aircraft siren human bark bird church music wind rain".split()


def runs(dates):
    dates = sorted(set(pd.Timestamp(d).normalize() for d in dates))
    result = []
    for d in dates:
        if result and d == result[-1][1] + pd.Timedelta(days=1):
            result[-1][1] = d
        else:
            result.append([d, d])
    return result


def ranges_text(dates):
    return "; ".join(a.strftime("%Y-%m-%d") if a == b else
                     f"{a:%Y-%m-%d} to {b:%Y-%m-%d}" for a, b in runs(dates))


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from the repository root so DuckDB views resolve.")
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect_analysis()
    checks = {}
    queries = {}

    def query(name, sql):
        queries[name] = sql
        print(name, flush=True)
        return con.sql(sql).df()

    # Numeric timestamp bounds cannot be truncated as string statistics can.
    loudness_pattern = (data_root()/"lake/scsm_loudness/date=*/sensor_id=*/*.parquet").as_posix().replace("'", "''")
    meta = query("raw_loudness_row_groups", f"""
        SELECT file_name, row_group_id, row_group_num_rows,
               regexp_extract(file_name, 'sensor_id=([^/\\\\]+)', 1) AS sensor_id,
               CAST(regexp_extract(file_name, 'date=([0-9-]+)', 1) AS DATE) AS date,
               CAST(stats_min_value AS TIMESTAMP_NS) AS first_time,
               CAST(stats_max_value AS TIMESTAMP_NS) AS last_time,
               stats_null_count
        FROM parquet_metadata('{loudness_pattern}')
        WHERE path_in_schema = 'time'
    """)
    assert meta.sensor_id.str.fullmatch(r"citizen-\d{3}").all()
    assert (meta.row_group_num_rows > 0).all()
    assert meta[['first_time', 'last_time', 'stats_null_count']].notna().all().all()
    assert (meta.stats_null_count == 0).all()
    assert (meta.first_time.dt.normalize() == meta.date).all()
    assert (meta.last_time.dt.normalize() == meta.date).all()
    checks['raw_time_statistics_complete_nonnull_and_within_partition_date'] = True
    checks['raw_parquet_files'] = int(meta.file_name.nunique())
    checks['raw_row_groups'] = len(meta)
    raw = meta.groupby(['sensor_id', 'date'], as_index=False).agg(
        rows=('row_group_num_rows', 'sum'), first_time=('first_time', 'min'),
        last_time=('last_time', 'max'))
    raw['dataset'] = 'spl_raw'
    # Validate sampling ratios against the published-data baseline.
    raw['sampling_ratio_within_daily_span'] = raw.rows / (
        8 * (raw.last_time - raw.first_time).dt.total_seconds())
    checks['spl_sampling_ratio_sensor_mean'] = float(
        raw.groupby('sensor_id').sampling_ratio_within_daily_span.mean().mean())
    checks['spl_sampling_ratio_sensor_min'] = float(
        raw.groupby('sensor_id').sampling_ratio_within_daily_span.mean().min())
    assert len(raw) == 2820, 'Unexpected number of sensor-days in the loudness data'
    assert abs(checks['spl_sampling_ratio_sensor_mean'] - 0.9476272839965063) < 1e-8

    valid = ' AND '.join(f'{c} IS NOT NULL' for c in CLASSES)
    sed_hourly = query("sed_hourly", f"""
        SELECT sensor_id, CAST(time AS DATE) AS date,
               date_trunc('hour', time) AS hour_start,
               COUNT(*) AS rows, MIN(time) AS first_time, MAX(time) AS last_time,
               COUNT(DISTINCT date_trunc('minute', time)) AS observed_minutes,
               COUNT(*) FILTER (WHERE time IS NULL) AS null_time_rows,
               COUNT(*) FILTER (WHERE CAST(time AS DATE) <> date) AS wrong_partition_rows,
               COUNT(*) FILTER (WHERE {valid}) AS all_classes_nonnull_rows
        FROM sources GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """)
    assert sed_hourly.null_time_rows.sum() == 0
    assert sed_hourly.wrong_partition_rows.sum() == 0
    sed = sed_hourly.groupby(['sensor_id', 'date'], as_index=False).agg(
        rows=('rows', 'sum'), first_time=('first_time', 'min'),
        last_time=('last_time', 'max'), observed_minutes=('observed_minutes', 'sum'),
        observed_hours=('hour_start', 'size'),
        all_classes_nonnull_rows=('all_classes_nonnull_rows', 'sum'))
    sed['dataset'] = 'sed_scores'
    assert (sed.rows == sed.all_classes_nonnull_rows).all()
    checks['sed_all_class_scores_nonnull'] = True

    mart = query("minute_loudness_daily", """
        SELECT sensor_id, CAST(time AS DATE) AS date, COUNT(*) AS rows,
               MIN(time) AS first_time, MAX(time) AS last_time,
               COUNT(DISTINCT time) AS observed_minutes,
               COUNT(DISTINCT date_trunc('hour', time)) AS observed_hours,
               COUNT(*) FILTER (WHERE time IS NULL) AS null_time_rows,
               COUNT(*) FILTER (WHERE CAST(time AS DATE) <> date) AS wrong_partition_rows,
               COUNT(*) FILTER (WHERE avg_spla > 0) AS positive_spla_rows
        FROM loudness_per_minute GROUP BY 1, 2 ORDER BY 1, 2
    """)
    assert mart.null_time_rows.sum() == 0
    assert mart.wrong_partition_rows.sum() == 0
    mart['dataset'] = 'spl_minute_mart'
    checks['mart_duplicate_minute_rows'] = int((mart.rows - mart.observed_minutes).sum())
    checks['mart_nonpositive_or_null_spla_rows'] = int((mart.rows-mart.positive_spla_rows).sum())
    keys = lambda df: set(zip(df.sensor_id, df.date))
    checks['raw_and_mart_sensor_dates_identical'] = keys(raw) == keys(mart)
    checks['sed_sensor_dates_subset_of_raw'] = keys(sed).issubset(keys(raw))
    assert checks['raw_and_mart_sensor_dates_identical']
    assert checks['sed_sensor_dates_subset_of_raw']

    obs = pd.read_csv(data_root() / 'ref/observation-timespans.csv', usecols=['sensor_id','start','end'])
    obs['sensor_id'] = obs.sensor_id.map(lambda n: f'citizen-{n:03d}')
    obs['start'] = pd.to_datetime(obs.start)
    obs['end'] = pd.to_datetime(obs.end)
    assert obs[['start','end']].notna().all().all()
    allowed = {sid: set().union(*(set(pd.date_range(r.start, r.end)) for r in g.itertuples()))
               for sid, g in obs.groupby('sensor_id')}
    for df in [raw, sed, mart]:
        assert all(d in allowed[s] for s, d in keys(df))
    checks['all_observed_dates_within_inclusive_reference_windows'] = True
    assert set(raw.sensor_id) == set(sed.sensor_id) == set(allowed)
    assert len(allowed) == 38 and 'citizen-033' not in allowed

    checks['sed_total_fragments'] = int(sed.rows.sum())
    assert checks['sed_total_fragments'] == 56923211

    # Compare sensor sets separately; matching sensors does not establish matching date coverage.
    table_sensors = {}
    for table in ['loudness_per_hour','ir_per_hour']:
        table_sensors[table] = con.sql(f'SELECT DISTINCT sensor_id FROM {table} ORDER BY 1').df().sensor_id.tolist()
    checks['materialized_table_sensor_sets_match_raw'] = all(set(s) == set(allowed) for s in table_sensors.values())

    daily = pd.concat([raw, sed, mart], ignore_index=True).sort_values(['sensor_id','dataset','date'])
    # Entire campaign calendar, including days with no records (explicit zeros).
    rows = []
    raw_idx = raw.set_index(['sensor_id','date'])
    sed_idx = sed.set_index(['sensor_id','date'])
    mart_idx = mart.set_index(['sensor_id','date'])
    all_dates = pd.date_range(raw.date.min(), raw.date.max())
    for sid in sorted(allowed):
        for date in all_dates:
            key = (sid,date)
            r = {'sensor_id':sid, 'date':date, 'within_reference_window':date in allowed[sid]}
            for label, index in [('spl',raw_idx),('sed',sed_idx),('spl_minute_mart',mart_idx)]:
                present = key in index.index
                r[f'{label}_observed'] = present
                r[f'{label}_rows'] = int(index.loc[key,'rows']) if present else 0
                if label != 'spl':
                    r[f'{label}_observed_minutes'] = int(index.loc[key,'observed_minutes']) if present else 0
            rows.append(r)
    calendar = pd.DataFrame(rows)
    intervals = []
    summaries = []
    for sid in sorted(allowed):
        row = {'sensor_id':sid, 'reference_windows':ranges_text(allowed[sid]),
               'reference_allowed_days':len(allowed[sid])}
        for label, df in [('spl',raw),('sed',sed)]:
            g = df[df.sensor_id == sid]
            dates = set(g.date)
            span = pd.date_range(g.date.min(),g.date.max())
            missing = set(span) - dates
            row.update({f'{label}_first_timestamp':g.first_time.min(),
                        f'{label}_last_timestamp':g.last_time.max(),
                        f'{label}_observed_days':len(dates),
                        f'{label}_calendar_span_days':len(span),
                        f'{label}_missing_days_within_span':len(missing),
                        f'{label}_observed_date_ranges':ranges_text(dates),
                        f'{label}_missing_date_ranges_within_span':ranges_text(missing),
                        f'{label}_rows':int(g.rows.sum())})
            for n,(a,b) in enumerate(runs(dates),start=1):
                intervals.append(dict(sensor_id=sid,dataset=label,run=n,start_date=a,
                                      end_date=b,observed_days=(b-a).days+1))
        r_dates = set(raw.loc[raw.sensor_id == sid,'date'])
        s_dates = set(sed.loc[sed.sensor_id == sid,'date'])
        row['spl_and_sed_overlap_days'] = len(r_dates & s_dates)
        row['spl_only_days'] = len(r_dates-s_dates)
        row['reference_days_without_spl'] = len(allowed[sid]-r_dates)
        row['reference_days_without_sed'] = len(allowed[sid]-s_dates)
        row['spl_mart_observed_minutes'] = int(mart.loc[mart.sensor_id == sid,'observed_minutes'].sum())
        row['sed_observed_minutes'] = int(sed.loc[sed.sensor_id == sid,'observed_minutes'].sum())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    monthly = daily.assign(month=daily.date.dt.strftime('%Y-%m')).groupby(
        ['sensor_id','dataset','month'],as_index=False).agg(
            observed_days=('date','size'),rows=('rows','sum'))
    sed_hourly['hour_of_day'] = sed_hourly.hour_start.dt.hour
    hod = sed_hourly.groupby(['sensor_id','hour_of_day'],as_index=False).agg(
        observed_dates=('date','nunique'), rows=('rows','sum'), observed_minutes=('observed_minutes','sum'))
    hod['share_of_sensor_sed_rows'] = hod.rows / hod.groupby('sensor_id').rows.transform('sum')
    common = []
    for label, df in [('spl',raw),('sed',sed)]:
        sets = [set(g.date) for _,g in df.groupby('sensor_id')]
        common.append({'dataset':label,'sensors':len(sets),
                       'date_ranges_with_records_at_all_sensors':ranges_text(set.intersection(*sets)),
                       'days_with_records_at_all_sensors':len(set.intersection(*sets))})
    campaign = calendar.groupby('date',as_index=False).agg(
        sensors_with_spl=('spl_observed','sum'), sensors_with_sed=('sed_observed','sum'),
        spl_rows=('spl_rows','sum'),sed_rows=('sed_rows','sum'))
    exports = {'sensor_coverage.csv':summary,'daily_coverage.csv':daily,
               'calendar_inclusion.csv':calendar,'observed_date_ranges.csv':pd.DataFrame(intervals),
               'monthly_coverage.csv':monthly,'sed_hour_of_day_coverage.csv':hod,
               'common_dates.csv':pd.DataFrame(common),'campaign_daily_coverage.csv':campaign}
    for name,df in exports.items():
        df.to_csv(OUT/name,index=False,encoding='utf-8',date_format='%Y-%m-%d %H:%M:%S.%f',float_format='%.9f')
    # Date-only columns should be straightforward to consume on any platform.
    for name,df in exports.items():
        for col in ['date','start_date','end_date']:
            if col in df:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        df.to_csv(OUT/name,index=False,encoding='utf-8',float_format='%.9f')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    duckdb_version=duckdb.__version__, database_opened_read_only=True,
                    checks=checks,queries=queries,
                    data_manifest_sha256=bundle_fingerprint(),
                    view_sql=dict(con.sql("SELECT view_name,sql FROM duckdb_views() WHERE NOT internal AND view_name IN ('loudness','sources','loudness_per_minute')").fetchall()))
    (OUT/'audit_checks.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)
    print(summary[['sensor_id','spl_observed_days','sed_observed_days']].to_string(index=False))
    print(pd.DataFrame(common).to_string(index=False))
    print('Summary statistics\n', summary[['spl_observed_days','sed_observed_days']].describe().to_string())
    con.close()


if __name__ == '__main__':
    main()
