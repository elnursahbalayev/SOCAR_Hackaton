"""Engineering report (one self-contained HTML page, printable to PDF) and a ZIP with every PTA plot.

Both are built from the ACTIVE results bundle, so they work for the provided data and for any upload.
Plots are drawn with matplotlib (same functions as the notebooks) and embedded as base64 PNG.
"""
import base64
import html
import io
import sys
import threading
import zipfile
from datetime import datetime
from functools import lru_cache

import matplotlib
matplotlib.use('Agg')                       # no screen on a server
import matplotlib.pyplot as plt             # noqa: E402
import pandas as pd                         # noqa: E402

from backend import data, pages             # noqa: E402

sys.path.insert(0, str(data.ROOT / 'src'))
import pta                                  # noqa: E402
import pta_model                            # noqa: E402

PLANNED = 'planned build-up test'
PLOT_LOCK = threading.Lock()          # matplotlib's pyplot is not thread-safe; API requests run in threads


def _png(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


def _rates_for_plots():
    r = data.daily_rates()
    return pd.DataFrame({'date': r.date, 'well_name': r.well_name, 'oil_rate_stbd': r.oil})


def pta_figures(result):
    """(build-up plots, model-match plot) as PNG bytes for one shut-in."""
    fig = pta.plot_event(data.high_freq(), result, _rates_for_plots())
    fig.suptitle(f"{result['well_name']} {result['t_shut']:%Y-%m-%d %H:%M} — {result['event_type']} "
                 f"({result['duration_hr']:.1f} h)", fontsize=13)
    buildup = _png(fig)
    match = _png(pta_model.plot_model_match(result, result['_fits'], result['_best'], result['_dt_fit'], result['_dp_fit']))
    return buildup, match


@lru_cache(maxsize=4)
def pta_plots_zip(bundle_id):
    """All PTA plots of a dataset, laid out like outputs/pta_plots/ (unplanned shut-ins in a sub-folder)."""
    buf = io.BytesIO()
    with PLOT_LOCK, zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for r in data.load_bundle(bundle_id)['pta']:
            folder = 'pta_plots/' if r['event_type'] == PLANNED else 'pta_plots/unplanned_shut_ins/'
            name = f"{r['well_name']}_{r['t_shut']:%Y%m%d}"
            buildup, match = pta_figures(r)
            z.writestr(folder + name + '_buildup.png', buildup)
            z.writestr(folder + name + '_model_match.png', match)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------------------------
def _img(png):
    return f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def _num(v, d=0):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '—'
    return f'{v:,.{d}f}'


def _table(head, rows):
    h = ''.join(f'<th>{html.escape(str(c))}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def _field_figures():
    rates = data.daily_rates()
    field = rates.groupby('date')[['oil', 'water', 'gas']].sum()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(field.index, field.oil, color='#1f9d55', label='oil [STB/d]')
    ax.plot(field.index, field.water, color='#2f6fdb', label='water [STB/d]')
    ax.plot(field.index, field.gas, color='#d64545', label='gas [Mscf/d]')
    ax.set_title('Field production (VFM, all wells)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    field_png = _png(fig)

    wells = sorted(rates.well_name.unique())
    tests = data.bundle()['tests']
    ncol = 5
    nrow = -(-len(wells) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.6 * nrow), squeeze=False)
    for ax, w in zip(axes.flat, wells):
        g = rates[rates.well_name == w]
        ax.plot(g.date, g.oil, color='#1f9d55', lw=1)
        if tests is not None:
            t = tests[tests.well_name == w]
            ax.plot(t.time, t.oil_rate_stbd, 'ko', ms=3)
        ax.set_title(w, fontsize=10)
        ax.tick_params(labelsize=7, axis='both')
        ax.tick_params(axis='x', rotation=30)
    for ax in list(axes.flat)[len(wells):]:
        ax.axis('off')
    fig.suptitle('Daily oil rate per well (line = VFM, dots = measured well tests) [STB/d]', fontsize=11)
    fig.tight_layout()
    return field_png, _png(fig)


CSS = """
body{font-family:Segoe UI,Calibri,Arial,sans-serif;color:#1c2430;margin:0;background:#fff}
main{max-width:1100px;margin:0 auto;padding:36px 40px}
h1{font-size:28px;margin:0}h2{font-size:20px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid #dde3ea}
h3{font-size:15px;margin:22px 0 6px}.muted{color:#5d6877}.small{font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}
th{background:#f2f5f8;text-align:left;font-weight:600}th,td{border:1px solid #dde3ea;padding:5px 8px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.kpi{border:1px solid #dde3ea;border-radius:8px;padding:10px 12px}.kpi b{display:block;font-size:20px}
img{max-width:100%;margin:8px 0}.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:#eef6f0;color:#1f7a45}
.warn{background:#fff4e0;color:#9a5b00}ul{margin:6px 0 6px 18px;padding:0}li{margin:3px 0}
@media print{main{padding:0}h2{break-after:avoid}.page{break-before:page}}
"""


@lru_cache(maxsize=4)
def report_html(bundle_id, oil_price=70.0):
    """The engineering report of the dataset `bundle_id` (call with data.ACTIVE['id'])."""
    with PLOT_LOCK:
        return _report_html(oil_price)


def _report_html(oil_price):
    info = data.dataset_info()
    ov = data.overview()
    vfm = pages.vfm_overview()
    ins = pages.insights(oil_price)
    results = data.pta_results()
    k = ov['kpis']
    field_png, wells_png = _field_figures()

    kpis = ''.join(f'<div class="kpi"><span class="muted small">{a}</span><b>{b}</b></div>' for a, b in [
        ('Oil (latest day)', f"{_num(k['oil'])} STB/d"), ('Water (latest day)', f"{_num(k['water'])} STB/d"),
        ('Gas (latest day)', f"{_num(k['gas'])} Mscf/d"), ('Water cut', f"{_num(k['water_cut'], 1)} %"),
        ('Cumulative oil in period', f"{_num(k['cum_oil_period'])} STB"),
        ('Shut-ins / build-up tests', f"{k['shut_in_events']} / {k['buildup_tests']}"),
        ('Oil deferred by shut-ins', f"{_num(k['deferred_oil_stb'])} STB"),
        ('Stimulation upside', f"{_num(ins['kpis']['stimulation_upside_stbd'])} STB/d")])

    val = vfm['validation']
    val_table = _table(['Fluid', 'Between tests WAPE %', 'Between tests MAPE %', 'Forecast 0–3 mo WAPE %', 'Forecast MAPE %'],
                       [[f, _num(val[f'{f}_wape'], 2), _num(val[f'{f}_mape'], 2), _num(val[f'forward_{f}_wape'], 2),
                         _num(val[f'forward_{f}_mape'], 2)] for f in ['oil', 'water', 'gas']])
    acc = info['accuracy_on_uploaded_tests']
    acc_html = ''
    if acc:
        acc_html = ('<h3>Accuracy on the well tests of this dataset</h3>' +
                    _table(['Fluid', 'R²', 'MAPE %', 'WAPE %'], [[f, _num(acc[f]['R2'], 3), _num(acc[f]['MAPE'], 2),
                                                               _num(acc[f]['WAPE'], 2)] for f in ['oil', 'water', 'gas']]) +
                    f'<p class="muted small">{acc["n_tests"]} tests; {acc.get("n_in_sample", 0)} of them inside the training '
                    'period (in-sample fit, not a blind check).</p>')
    well_rows = [[w['well'], _num(w['oil']), _num(w['water']), _num(w['gas']), _num(w['water_cut'], 1), _num(w['p_res']),
                  _num(w['p_res_slope_psi_month'], 1), _num(w['J'], 2), w['J_source'] or '—', _num(w['k_md'], 1),
                  _num(w['skin'], 2), _num(w['cum_oil'])] for w in vfm['wells']]
    well_table = _table(['Well', 'Oil STB/d', 'Water STB/d', 'Gas Mscf/d', 'WC %', 'P_res psia', 'dP_res/dt psi/mo',
                         'J STB/d/psi', 'J source', 'k md', 'Skin', 'Cum oil STB'], well_rows)

    planned = [r for r in results if r['event_type'] == PLANNED]
    other = [r for r in results if r['event_type'] != PLANNED]
    pta_table = _table(['Well', 'Shut-in', 'Duration h', 'q STB/d', 'Horner m psi/cycle', 'k md', 'Skin', 'ΔP skin psi',
                        'Flow eff.', 'P* psia', 'Boundary'],
                       [[r['well_name'], f"{r['t_shut']:%Y-%m-%d %H:%M}", _num(r['duration_hr'], 1), _num(r['q_oil_stbd']),
                         _num(r['m_psi_cycle'], 2), _num(r['k_md'], 2), _num(r['skin'], 2), _num(r['dP_skin_psi'], 2),
                         _num(r['flow_efficiency'], 2), _num(r['p_star_psia']), r['model_boundary'].replace('_', ' ')]
                        for r in planned]) if planned else '<p>No planned build-up test in this dataset.</p>'
    other_table = _table(['Well', 'Shut-in', 'Duration h', 'Why not a build-up test', 'k md (info)', 'Skin (info)'],
                         [[r['well_name'], f"{r['t_shut']:%Y-%m-%d %H:%M}", _num(r['duration_hr'], 1),
                           html.escape(r['classification_reason']), _num(r['k_md'], 1), _num(r['skin'], 2)] for r in other]) if other else ''
    pta_plots = ''
    for r in planned:
        buildup, match = pta_figures(r)
        pta_plots += (f'<h3 class="page">{r["well_name"]} — {r["t_shut"]:%Y-%m-%d %H:%M} ({r["duration_hr"]:.0f} h)</h3>'
                      f'<p>{html.escape(r["interpretation"])}</p>{_img(buildup)}' + _img(match).replace("<img", "<img style=\"max-width:640px\"", 1))

    actions = []
    for s in ins['stimulation']:
        actions.append(f"<li><b>{s['well']}: stimulation candidate</b> — skin {s['skin']:.1f}, flow efficiency "
                       f"{s['flow_efficiency']:.2f}; removing the skin ≈ +{s['upside_stbd']:.0f} STB/d "
                       f"(≈ ${s['value_usd_year'] / 1e6:.1f} M/year at ${oil_price:.0f}/bbl) "
                       f"<span class='badge {'' if s['planned'] else 'warn'}'>{s['confidence']}</span></li>")
    if ins['lift']:
        worst = ins['lift'][-1]
        actions.append(f"<li><b>{worst['well']}: review lift-gas allocation</b> — {worst['lift_gas_mscfd']:.0f} Mscf/d of lift gas for "
                       f"{worst['oil_stbd']:.0f} STB/d ({worst['oil_per_mscf']:.2f} STB per Mscf, {worst['vs_median_pct']:.0f} % vs field median)</li>")
    for h in ins['health']:
        if h['flags']:
            actions.append(f"<li><b>{h['well']}: {', '.join(h['flags'])}</b> — P_res {_num(h['p_res'])} psia "
                           f"({_num(h['p_res_slope_psi_month'], 1)} psi/month), water cut {_num(h['water_cut'], 1)} % "
                           f"({_num(h['wc_slope_pp_month'], 2)} pp/month)</li>")
    if ins['gaps']:
        actions.append('<li><b>Schedule build-up tests</b> (k and skin unknown, highest rate first): ' +
                       ', '.join(f"{g['well']} ({g['oil_stbd']:.0f} STB/d)" for g in ins['gaps']) + '</li>')

    cr = info['clean_report']
    quality = _table(['Sensor', 'Null codes (−999.25)', 'Spikes removed'],
                     [[s, _num(cr['null_codes'][s]), _num(cr['spikes_removed'][s])] for s in cr['null_codes']])
    files = _table(['File', 'Recognised as', 'Rows', 'Period'],
                   [[html.escape(f['file']), f['type'] or 'not recognised', _num(f.get('rows')),
                     f"{f['start'][:10]} → {f['end'][:10]}" if f.get('start') else 'static'] for f in info['files']])
    warn = ''.join(f'<li>{html.escape(w)}</li>' for w in info['warnings']) or '<li>none</li>'

    body = f"""
<main>
<h1>WellMind — engineering report</h1>
<p class="muted">{info['label']} · {info['period']['start'][:10]} → {info['period']['end'][:10]} · {len(info['wells'])} wells ·
generated {datetime.now():%Y-%m-%d %H:%M} · oil price ${oil_price:.0f}/bbl</p>
<div class="kpis">{kpis}</div>

<h2>1. Recommended actions</h2><ul>{''.join(actions) or '<li>No action flagged.</li>'}</ul>

<h2>2. Virtual flow meter</h2>
<p><b>Method.</b> Oil: q<sub>o</sub> = J·(P<sub>res</sub>(t) − P<sub>wf</sub>), P<sub>wf</sub> from the downhole gauge,
J from Darcy with our own build-up k and skin (or fitted from tests), P<sub>res</sub>(t) a local linear trend per well.
Gas: q<sub>g</sub> = q<sub>o</sub>·R<sub>s</sub>. Water: ½ physics water-cut trend + ½ Gaussian process on
separator-safe inputs. Daily rate = flowing rate × fraction of the day on stream. Model: {html.escape(vfm['model']['name'])},
trained on well tests up to {vfm['model']['trained_until'][:10]}.</p>
<h3>Validation of the stored model (official metrics)</h3>{val_table}{acc_html}
{_img(field_png)}{_img(wells_png)}
<h3>Per-well summary (latest day)</h3>{well_table}

<h2 class="page">3. Pressure transient analysis</h2>
<p><b>Method.</b> Shut-ins detected from the choke (0 %), classified as planned build-up tests (≥ 24 h and a staged
restart) or unplanned shut-ins. Radial-flow window = flattest Bourdet derivative; Horner fit
P<sub>ws</sub> = P* − m·log((t<sub>p</sub>+Δt)/Δt); k = 162.6·q·B·μ/(m·h);
s = 1.151·[(P<sub>1hr</sub> − P<sub>wf</sub>)/m − log(k/(φμc<sub>t</sub>r<sub>w</sub>²)) + 3.23]; ΔP<sub>skin</sub> = 0.869·m·s.
q = VFM oil rate (mean of the 7 days before), cross-checked with analytical model matching.</p>
<h3>Build-up tests (deliverable)</h3>{pta_table}
{'<h3>Unplanned shut-ins (information only)</h3>' + other_table if other_table else ''}
{pta_plots}

<h2 class="page">4. Data quality</h2>{files}{quality}
<p class="small muted">Duplicate rows dropped: {_num(cr['duplicates_dropped'])} · shut-in minutes: {_num(cr['shut_in_minutes'])}</p>
<h3>Warnings</h3><ul>{warn}</ul>
</main>"""
    return f'<!doctype html><html><head><meta charset="utf-8"><title>WellMind report</title><style>{CSS}</style></head><body>{body}</body></html>'


def warm_up():
    """Pre-compute the slow parts for the ACTIVE dataset (bundle load, overview, report, plot ZIP) so that
    every page opens instantly during a demo. Runs in a background thread; errors are only printed."""
    try:
        bundle_id = data.ACTIVE['id']
        data.overview()
        pages.vfm_overview()
        pages.insights(70.0)
        report_html(bundle_id, 70.0)
        pta_plots_zip(bundle_id)
        print(f'warm-up done for {bundle_id}', flush=True)
    except Exception as exc:          # never block the server because of a warm-up problem
        print('warm-up failed:', exc, flush=True)
