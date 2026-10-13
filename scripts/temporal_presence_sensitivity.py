"""Pooled temporal presence sensitivity; run from repository root.
Reuses verified minute aggregates and independently queries raw source period totals.
Writes only out/temporal_presence_sensitivity and figures/temporal_presence_sensitivity.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_paths import connect_analysis, output_path, bundle_fingerprint
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
CLASSES=['vehicle','honking','aircraft','siren','human','bark','bird','church','music','wind','rain']
THRESHOLDS=[.3,.5,.7]
NAMES=dict(zip(CLASSES,['Vehicles','Honking','Aircraft','Sirens','Human speech / crowd','Dogs','Bird song','Church bells','Music','Wind','Rain']))
COLORS={.3:'#0072B2',.5:'#333333',.7:'#D55E00'}

def period_of(minute):
    minute=np.asarray(minute)
    return np.select([(minute>=420)&(minute<1140),(minute>=1140)&(minute<1380)],['day','evening'],default='night')

def period_sql(classes=CLASSES,thresholds=THRESHOLDS):
    fields=['COUNT(*) AS n_fragments','COUNT(DISTINCT sensor_id) AS n_sensors']
    for cls in classes:
        if cls not in CLASSES: raise ValueError(cls)
        fields.append(f'COUNT(*) FILTER (WHERE "{cls}" IS NULL) AS {cls}_nulls')
        for t in thresholds:
            if t not in THRESHOLDS: raise ValueError(t)
            fields.append(f'COUNT(*) FILTER (WHERE "{cls}" > {t}) AS {cls}_{int(t*10)}')
    return ("SELECT CASE WHEN EXTRACT(hour FROM time)>=7 AND EXTRACT(hour FROM time)<19 THEN 'day' "
            "WHEN EXTRACT(hour FROM time)>=19 AND EXTRACT(hour FROM time)<23 THEN 'evening' ELSE 'night' END AS period, "
            +', '.join(fields)+' FROM sources WHERE time IS NOT NULL GROUP BY 1 ORDER BY 1')

def main():
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('Run from repository root.')
    out=output_path('out/temporal_presence_sensitivity'); figs=output_path('figures/temporal_presence_sensitivity')
    out.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    checks={}
    def check(label,condition):
        checks[label]=bool(condition)
        if not condition: raise AssertionError(label)
    inputs=[output_path('out/diurnal_presence/threshold_curves.csv'),output_path('out/diurnal_presence/period_summary.csv'),output_path('out/diurnal_presence/verification_manifest.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    previous=json.loads(inputs[2].read_text(encoding='utf-8'))
    curves=pd.read_csv(inputs[0]); prior_period=pd.read_csv(inputs[1])
    check('33 complete 1440-minute grids',len(curves)==11*3*1440 and not curves.duplicated(['sound_class','threshold','minute_of_day']).any())
    print('Fresh raw-data period aggregation, eleven classes at three cutoffs',flush=True)
    query=period_sql()
    with connect_analysis() as db:
        view=db.execute("SELECT sql FROM duckdb_views() WHERE view_name='sources'").fetchone()[0]
        raw=db.execute(query).df().set_index('period')
    raw.to_csv(out/'raw_period_counts.csv')
    check('raw period selection totals 56923211 fragments',raw.n_fragments.sum()==56923211)
    check('38 sensors in each period',raw.n_sensors.eq(38).all())
    check('same prepared-data bundle',bundle_fingerprint()==previous['data_manifest_sha256'])
    periods=[]; shapes=[]; contrasts=[]; peaks=[]
    for (cls,t),g in curves.groupby(['sound_class','threshold']):
        g=g.sort_values('minute_of_day').copy()
        tag=f'{cls}/{t}'
        check(tag+': observed minute range',g.minute_of_day.tolist()==list(range(1440)))
        check(tag+': positive denominators',g.n_fragments.gt(0).all())
        check(tag+': valid hit bounds',g.detections.between(0,g.n_fragments).all())
        check(tag+': exact pooled ratio',np.allclose(g.presence_pct,100*g.detections/g.n_fragments,rtol=0,atol=1e-12))
        smooth=g.presence_pct.rolling(60,center=True,min_periods=20).mean()
        check(tag+': rolling mean',np.allclose(g.presence_pct_smooth,smooth,rtol=0,atol=1e-12))
        g['period']=period_of(g.minute_of_day)
        period_values={}
        for p,z in g.groupby('period'):
            hits=int(z.detections.sum()); total=int(z.n_fragments.sum())
            check(tag+'/'+p+': independent raw numerator',hits==raw.loc[p,f'{cls}_{int(t*10)}'])
            check(tag+'/'+p+': independent raw denominator',total==raw.loc[p,'n_fragments'])
            check(tag+'/'+p+': no missing scores',raw.loc[p,cls+'_nulls']==0)
            pct=100*hits/total; period_values[p]=pct
            periods.append(dict(sound_class=cls,threshold=t,period=p,detections=hits,n_fragments=total,
                pooled_presence_pct=pct,mean_minute_presence_pct=z.presence_pct.mean()))
        contrasts.append(dict(sound_class=cls,threshold=t,day_pct=period_values['day'],evening_pct=period_values['evening'],
            night_pct=period_values['night'],night_minus_day_pp=period_values['night']-period_values['day'],
            night_day_ratio=period_values['night']/period_values['day'] if period_values['day'] else np.nan))
        baseline=curves[curves.sound_class.eq(cls)&curves.threshold.eq(.5)].sort_values('minute_of_day')
        shapes.append(dict(sound_class=cls,threshold=t,pearson_minute_vs_05=pearsonr(g.presence_pct,baseline.presence_pct).statistic,
            spearman_minute_vs_05=spearmanr(g.presence_pct,baseline.presence_pct).statistic,
            pearson_smoothed_vs_05=pearsonr(smooth,baseline.presence_pct_smooth).statistic))
        for label,mask in [('whole_day',np.ones(len(g),dtype=bool)),('morning_00_12',g.minute_of_day.lt(720)),('late_day_16_24',g.minute_of_day.ge(960))]:
            z=g.loc[mask]; k=z.presence_pct_smooth.idxmax(); minute=int(g.loc[k,'minute_of_day'])
            peaks.append(dict(sound_class=cls,threshold=t,window=label,smoothed_peak_time=f'{minute//60:02d}:{minute%60:02d}',
                peak_minute=minute,presence_pct_smooth=g.loc[k,'presence_pct_smooth']))
    periods=pd.DataFrame(periods); contrasts=pd.DataFrame(contrasts); shapes=pd.DataFrame(shapes); peaks=pd.DataFrame(peaks)
    keys=['sound_class','threshold','period']
    check('periods reproduce earlier percentage export',np.allclose(periods.set_index(keys).sort_index().pooled_presence_pct,
        prior_period.set_index(keys).sort_index().pooled_presence_pct,rtol=0,atol=1e-12))
    flags=[]
    for cls,g in contrasts.groupby('sound_class'):
        g=g.set_index('threshold').sort_index(); differences=g.night_minus_day_pp
        flags.append(dict(sound_class=cls,sign_reversal=bool(len(set(np.sign(differences)))>1),
            absolute_contrast_decreases_each_step=bool(np.all(np.diff(abs(differences))<0)),
            night_higher_at_all_thresholds=bool(differences.gt(0).all())))
    flags=pd.DataFrame(flags)
    for name,frame in [('period_summary',periods),('day_night_contrasts',contrasts),('curve_similarity',shapes),('peak_times',peaks),('contrast_flags',flags)]:
        frame.to_csv(out/(name+'.csv'),index=False)
    curves.to_csv(out/'threshold_curves.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained',gridspec_kw={'height_ratios':[1.5,1]})
    for start,end,color in [(0,7,'#e9e9f5'),(7,19,'#fff8dd'),(19,23,'#fff0dd'),(23,24,'#e9e9f5')]: axes[0].axvspan(start,end,color=color,zorder=0)
    for j,t in enumerate(THRESHOLDS):
        g=curves[curves.sound_class.eq('vehicle')&curves.threshold.eq(t)].sort_values('minute_of_day')
        axes[0].plot(g.minute_of_day/60,g.presence_pct,color=COLORS[t],alpha=.22,lw=.6)
        axes[0].plot(g.minute_of_day/60,g.presence_pct_smooth,color=COLORS[t],lw=1.8,label=f'Cutoff {t}')
        p=periods[periods.sound_class.eq('vehicle')&periods.threshold.eq(t)].set_index('period').loc[['day','evening','night']]
        bars=axes[1].bar(np.arange(3)+(j-1)*.24,p.pooled_presence_pct,width=.23,color=COLORS[t],label=f'Cutoff {t}')
        axes[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[0].set(xlim=(0,24),ylim=(0,100),ylabel='Traffic presence (%)',xlabel='Stored time of day',title='Minute-level pooled ratios; bold lines: centred 60-minute means')
    axes[0].set_xticks(range(0,25,4),[f'{h:02d}:00' for h in range(0,25,4)]); axes[0].legend(ncol=3)
    axes[1].set(ylim=(0,90),ylabel='Traffic presence (%)',title='Period ratios: pooled detections / pooled recorded fragments')
    axes[1].set_xticks(range(3),['Day (07:00–19:00)','Evening (19:00–23:00)','Night (23:00–07:00)'])
    for ax in axes: ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Traffic temporal sensitivity to the detection cutoff',fontsize=14)
    for ext in ['png','svg']: fig.savefig(figs/f'traffic_temporal_sensitivity.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(12,12),layout='constrained')
    for cls,ax in zip(CLASSES,axes.flat):
        for t in THRESHOLDS:
            g=curves[curves.sound_class.eq(cls)&curves.threshold.eq(t)].sort_values('minute_of_day')
            ax.plot(g.minute_of_day/60,g.presence_pct_smooth,color=COLORS[t],label=f'{t}',lw=1.4)
        ax.set(title=NAMES[cls],xlim=(0,24),ylim=(0,None),ylabel='Presence (%)')
        ax.set_xticks([0,6,12,18,24],['00','06','12','18','24']); ax.grid(alpha=.2)
    axes.flat[-1].axis('off'); axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),title='Detection cutoff',loc='center',frameon=False)
    fig.suptitle('All-class temporal sensitivity\nPooled minute percentages; centred 60-minute means; independent vertical scales',fontsize=14)
    for ext in ['png','svg']: fig.savefig(figs/f'all_class_temporal_sensitivity.{ext}',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    y=np.arange(len(CLASSES))
    for j,t in enumerate(THRESHOLDS):
        g=contrasts[contrasts.threshold.eq(t)].set_index('sound_class').loc[CLASSES]
        ax.barh(y+(j-1)*.24,g.night_minus_day_pp,height=.23,color=COLORS[t],label=f'Cutoff {t}')
    ax.axvline(0,color='black',lw=.8); ax.set_yticks(y,[NAMES[c] for c in CLASSES]); ax.invert_yaxis()
    ax.set(xlabel='Night minus day presence (percentage points)',title='Direction of the day–night contrast across thresholds')
    ax.legend(); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    for ext in ['png','svg']: fig.savefig(figs/f'all_class_day_night_contrasts.{ext}',dpi=170)
    plt.close(fig)
    for f,digest in hashes.items(): check('unchanged input '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
    manifest=dict(generated_utc=datetime.now(timezone.utc).isoformat(),script='scripts/temporal_presence_sensitivity.py',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.executable,duckdb=duckdb.__version__,
        execution='Minute curves from regenerate_diurnal_plots.py; period numerators and denominators independently recomputed from the prepared source lake.',
        source_view_sql=view,period_sql=query,input_sha256=hashes,checks=checks,
        selection='38 sensors, citizen-033 excluded by existing source lake; all timestamped source fragments, no loudness filter; retained deployment windows',
        denominator='All timestamped fragments, including NULL scores; NULLs are not positive detections; observed missing score counts zero',
        comparison='score > threshold, individually per class; thresholds .3/.5/.7; no argmax',
        weighting='Fragment-pooled ratios by minute and by period; equally weighted mean-minute percentages exported separately',
        time='Stored naive clock, no added timezone correction; day [07:00,19:00), evening [19:00,23:00), night otherwise',
        smoothing='Centred 60-point mean, minimum 20; all 1440 minutes observed; truncated at midnight; not used for period ratios',
        model_provenance='Existing deployed scores; Deployment model documentation identifies 20250411-14-38-24-ft-default-mel104-fmax8000 quantized-4, model SHA-256 46fa012420f46d5cecbf3a3fe09e77024b1d9246741e27fd802bc1925133f985. Model identity reported from prior work, not revalidated here; no inference rerun.')
    (out/'verification_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'{len(checks)} checks passed',flush=True)
    print(contrasts[contrasts.sound_class.eq('vehicle')].to_string(index=False))
    print(flags.to_string(index=False))
    print('Curve correlations against .5:'); print(shapes[shapes.threshold.ne(.5)].to_string(index=False))
if __name__=='__main__': main()
