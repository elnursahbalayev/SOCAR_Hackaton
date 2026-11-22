"""Write the deliverable Excel files (also used by the UI's export buttons)."""
import pandas as pd

VFM_COLUMNS = ['date', 'well_name', 'oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd']


def vfm_table(daily, rates):
    """The required deliverable table: date, well_name, oil_rate_stbd, water_rate_stbd, gas_rate_mscfd."""
    out = pd.DataFrame({'date': daily.date.dt.date, 'well_name': daily.well_name})
    for col in ['oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd']:
        out[col] = rates[col].round(2)
    return out[VFM_COLUMNS].sort_values(['well_name', 'date']).reset_index(drop=True)


def write_vfm_excel(path, predictions, model_info, validation=None, details=None):
    """Sheet 1 'predictions' = exactly the required columns. Extra sheets document the model."""
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        predictions.to_excel(xl, sheet_name='predictions', index=False)
        pd.DataFrame(model_info, columns=['item', 'value']).to_excel(xl, sheet_name='model_info', index=False)
        if validation is not None:
            validation.to_excel(xl, sheet_name='validation')
        if details is not None:
            details.to_excel(xl, sheet_name='daily_details', index=False)
        for ws in xl.book.worksheets:                       # readable column widths
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 80)


PTA_COLUMNS = {                      # required deliverable columns (internal name -> Excel header)
    'well_name': 'well_name',
    't_shut': 'shut_in_start',
    'duration_hr': 'shut_in_duration_hr',
    'm_psi_cycle': 'horner_slope_m_psi_per_cycle',
    'skin': 'skin',
    'k_md': 'permeability_md',
    'dP_skin_psi': 'delta_p_skin_psi',
}


def pta_table(results):
    """The required PTA deliverable: one row per shut-in with Horner slope, skin, k (md), dP skin (psi)."""
    out = results[list(PTA_COLUMNS)].rename(columns=PTA_COLUMNS)
    return out.round({'shut_in_duration_hr': 2, 'horner_slope_m_psi_per_cycle': 2, 'skin': 2,
                      'permeability_md': 2, 'delta_p_skin_psi': 2})


def write_pta_excel(path, results_table, details=None, method=None, sensitivity=None):
    """Sheet 1 'pta_results' = required columns. Extra sheets: details, method (formulas), sensitivity."""
    sheets = {'pta_results': results_table, 'pta_details': details, 'sensitivity': sensitivity,
              'method': pd.DataFrame(method, columns=['item', 'value']) if method else None}
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        for name, df in sheets.items():
            if df is not None:
                df.to_excel(xl, sheet_name=name, index=False)
        for ws in xl.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 90)


# ---------------------------------------------------------------------------------------------
# Benchmark submission files (exact column names / order of the organisers' templates)
# ---------------------------------------------------------------------------------------------
VFM_SUBMISSION_COLUMNS = ['timestamp', 'well_name', 'oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd']
PTA_SUBMISSION_COLUMNS = ['well_name', 'shut_in_time', 'permeability_md', 'skin', 'delta_p_skin_psi', 'horner_slope']


def vfm_submission(vfm_predictions, timestamp_format='%Y-%m-%d'):
    """vfm_submission_template.csv: one row per well per day (daily rates in STB/d and Mscf/d)."""
    out = vfm_predictions.copy()
    out['timestamp'] = pd.to_datetime(out['date']).dt.strftime(timestamp_format)
    return out[VFM_SUBMISSION_COLUMNS].round(2)


def pta_submission(pta_results):
    """pta_submission_template.csv: one row per shut-in. pta_results = the 'pta_results' sheet of PTA_results.xlsx."""
    out = pd.DataFrame({
        'well_name': pta_results['well_name'],
        'shut_in_time': pd.to_datetime(pta_results['shut_in_start']).dt.strftime('%Y-%m-%d %H:%M:%S'),
        'permeability_md': pta_results['permeability_md'],
        'skin': pta_results['skin'],
        'delta_p_skin_psi': pta_results['delta_p_skin_psi'],
        'horner_slope': pta_results['horner_slope_m_psi_per_cycle'],
    })
    return out[PTA_SUBMISSION_COLUMNS]


def write_submission(df, path):
    """Write a submission CSV exactly like the templates: comma-separated, no index, no BOM, LF line endings."""
    df.to_csv(path, index=False, lineterminator='\n', encoding='utf-8')
