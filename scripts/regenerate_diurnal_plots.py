"""Refresh diurnal percentage outputs; run from repository root.
Writes diurnal_presence exports; reads notebook parameters without modifying cells.
Outputs contain numerical summaries, figures and reproducibility checks.
"""
from pathlib import Path
import hashlib, json, sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import duckdb
from analysis_paths import connect_analysis, database_path, output_path, bundle_fingerprint
import numpy as np
import pandas as pd
import helpers
CLASSES = ['vehicle','honking','aircraft','siren','human','bark','bird','church','music','wind','rain']
THRESHOLDS = [.3,.5,.7]

def main():
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('Run from repository root.')
    out, figures=output_path('out/diurnal_presence'), output_path('figures/diurnal_presence')
    out.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    path=ROOT/'analysis_2_sources.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [database_path(), *ROOT.glob('analysis_*.ipynb')]}
    checks={}
    def check(name, condition):
        checks[name]=bool(condition)
        if not condition: raise AssertionError(name)
    print('Fresh pooled SQL: eleven classes and three thresholds',flush=True)
    parts=['COUNT(*) AS n_fragments','COUNT(DISTINCT date) AS observed_dates']
    for cls in CLASSES:
        parts.append(f'COUNT(*) FILTER (WHERE "{cls}" IS NULL) AS {cls}_nulls')
        for t in THRESHOLDS: parts.append(f'COUNT(*) FILTER (WHERE "{cls}" > {t}) AS {cls}_{int(t*10)}')
    with connect_analysis() as db:
        view=db.execute("SELECT sql FROM duckdb_views() WHERE view_name='sources'").fetchone()[0]
        direct=db.execute('SELECT CAST(EXTRACT(hour FROM time)*60+EXTRACT(minute FROM time) AS INTEGER) AS minute_of_day, '+
            ', '.join(parts)+' FROM sources WHERE time IS NOT NULL GROUP BY 1 ORDER BY 1').df()
        audit=db.sql('SELECT sensor_id, COUNT(*) AS n_fragments, COUNT(*) FILTER (WHERE time IS NULL) AS null_time, '
                     'MIN(time) AS first_time, MAX(time) AS last_time FROM sources GROUP BY sensor_id ORDER BY sensor_id').df()
    direct.to_csv(out/'direct_minute_counts.csv',index=False); audit.to_csv(out/'sensor_audit.csv',index=False)
    check('all minutes observed',direct.minute_of_day.tolist()==list(range(1440)))
    check('38 sensors excluding 033',len(audit)==38 and 'citizen-033' not in set(audit.sensor_id))
    check('no null timestamps',audit.null_time.sum()==0)
    check('all 56923211 intervals reconcile',direct.n_fragments.sum()==audit.n_fragments.sum()==56923211)
    labels=np.select([(direct.minute_of_day>=420)&(direct.minute_of_day<1140),
                      (direct.minute_of_day>=1140)&(direct.minute_of_day<1380)],['day','evening'],default='night')
    frames, periods, shape=[],[],[]
    for cls in CLASSES:
        check(cls+': no null scores',direct[cls+'_nulls'].sum()==0)
        for t in THRESHOLDS:
            hits=direct[f'{cls}_{int(t*10)}']
            g=pd.DataFrame({'sound_class':cls,'threshold':t,'minute_of_day':direct.minute_of_day,
                            'detections':hits,'n_fragments':direct.n_fragments})
            g['presence_pct']=100*g.detections/g.n_fragments
            g['presence_pct_smooth']=g.presence_pct.rolling(60,center=True,min_periods=20).mean()
            check(f'{cls}/{t}: bounds',g.presence_pct.between(0,100).all()); frames.append(g)
            for period in ['day','evening','night']:
                p=g.loc[labels==period]
                periods.append(dict(sound_class=cls,threshold=t,period=period,detections=int(p.detections.sum()),
                    n_fragments=int(p.n_fragments.sum()),pooled_presence_pct=100*p.detections.sum()/p.n_fragments.sum(),
                    mean_minute_presence_pct=p.presence_pct.mean()))
            if t==.5:
                peak=int(g.loc[g.presence_pct_smooth.idxmax(),'minute_of_day'])
                shape.append(dict(sound_class=cls,min_pct=g.presence_pct.min(),max_pct=g.presence_pct.max(),
                    smoothed_peak_time=f'{peak//60:02d}:{peak%60:02d}'))
    curves=pd.concat(frames,ignore_index=True); curves.to_csv(out/'threshold_curves.csv',index=False)
    pd.DataFrame(periods).to_csv(out/'period_summary.csv',index=False); pd.DataFrame(shape).to_csv(out/'curve_summary.csv',index=False)
    for cls in CLASSES:
        g=curves[curves.sound_class.eq(cls)].pivot(index='minute_of_day',columns='threshold',values='presence_pct')
        check(cls+': threshold monotonicity',((g[.3]>=g[.5])&(g[.5]>=g[.7])).all())
    def capture(source,**kwargs):
        return_data=kwargs.get('return_df',False); kwargs.update(show=False,return_df=True)
        fig,data=helpers.plot_source_avg_per_minute(source,**kwargs)
        expected=curves[curves.sound_class.eq(source)&curves.threshold.eq(.5)]
        for col in ['detections','n_fragments','presence_pct','presence_pct_smooth']:
            check(source+': independent '+col,np.allclose(data[col],expected[col],rtol=1e-12,atol=1e-12,equal_nan=True))
        check(source+': plotted percentages',np.allclose(np.asarray(fig.data[0].y),expected.presence_pct))
        check(source+': plotted smoothing',np.allclose(np.asarray(fig.data[-1].y),expected.presence_pct_smooth))
        data.to_csv(out/f'{source}_diurnal.csv',index=False)
        fig.write_html(str(figures/f'{source}_diurnal.html'),include_plotlyjs=True)
        return (fig,data) if return_data else fig
    for i,cell in enumerate(nb['cells']):
        source=''.join(cell['source'])
        if cell['cell_type']=='code' and 'plot_source_avg_per_minute(' in source and 'SOURCE = ' in source:
            ns={'plot_source_avg_per_minute':capture}; exec(compile(source,str(path),'exec'),ns)
            print('Verified and exported',ns['SOURCE'],flush=True)
    for f,digest in protected.items(): check('unchanged '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
    manifest=dict(generated_utc=datetime.now(timezone.utc).isoformat(),execution='fresh lake aggregation and live helper execution',
        duckdb=duckdb.__version__,source_view_sql=view,data_manifest_sha256=bundle_fingerprint(),
        ratio='100 * COUNT(score > cutoff) / COUNT(all timestamped fragments), pooled by minute of day',
        selection='existing sources view, time IS NOT NULL; 38 sensors excluding citizen-033',
        missing_scores='included in denominator, not positive detections; actual NULL count zero',
        smoothing='centred 60-minute mean of minute percentages; minimum 20 points; no midnight wrap',
        periods='stored-clock day 07:00-19:00, evening 19:00-23:00, night 23:00-07:00; no new timezone offset',
        period_weighting='pooled_presence_pct pools fragments; mean_minute_presence_pct equally weights minute ratios',
        checks=checks,notebooks_modified=False,code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
            for f in ['helpers.py','scripts/regenerate_diurnal_plots.py']})
    (out/'verification_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'{len(checks)} checks passed. Individual plots: {figures}',flush=True)
    print(pd.DataFrame(periods).query("sound_class == 'vehicle'").to_string(index=False),flush=True)
if __name__=='__main__': main()
