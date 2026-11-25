"""Aggregate the 2024 steel monthly enterprise files and run Figure4 analysis.

Usage:
python figure4_steel_enterprise_from_monthly.py
python figure4_steel_enterprise_from_monthly.py --source-dir <summary_2024> --output-dir <output>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from figure4_concentration import signature, write_csv
from figure4_steel_enterprise import main as analyze_enterprises

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = Path(
    r'E:\monthly_points_activity_steel\emission\02a_emission_fact\summary_2024'
)
DEFAULT_OUTPUT_DIR = HERE / 'steel_enterprise_analysis'
MONTHS = [f'2024{month:02d}' for month in range(1, 13)]
MONTHLY_PREFIX = '.enterprise_standard_'
REQUIRED_COLUMNS = [
    'enterprise_id',
    'standard',
    '类别',
    'CO2_kg',
    'distance_km',
    'CO2_speed_added_kg',
    'distance_added_km',
]
RECONCILE_COLUMNS = [
    'CO2_kg',
    'distance_km',
    'CO2_speed_added_kg',
    'distance_added_km',
]


def read_monthly_files(source_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = [source_dir / f'{MONTHLY_PREFIX}{month}.csv' for month in MONTHS]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Missing required monthly source files in {source_dir}: {", ".join(missing)}'
        )

    frames = []
    for month, path in zip(MONTHS, files):
        data = pd.read_csv(
            path,
            dtype={'enterprise_id': 'string'},
            float_precision='round_trip',
        )
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
        if missing_columns:
            raise ValueError(
                f'{path.name} is missing columns: {", ".join(missing_columns)}'
            )
        if data[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(f'{path.name} has blank required values.')
        data['enterprise_id'] = data.enterprise_id.str.strip()
        if data.enterprise_id.eq('').any():
            raise ValueError(f'{path.name} has blank enterprise_id values.')
        duplicate_keys = data.duplicated(
            ['enterprise_id', 'standard', '类别']
        )
        if duplicate_keys.any():
            raise ValueError(
                f'{path.name} has {int(duplicate_keys.sum())} duplicate '
                'enterprise/standard/category records.'
            )
        for column in RECONCILE_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors='raise')
            values = data[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(
                    f'{path.name}: {column} must be finite and non-negative.'
                )
        data['_source_month'] = month
        frames.append(data)
    return pd.concat(frames, ignore_index=True), files


def check_annual_reconciliation(
    all_months: pd.DataFrame,
    source_dir: Path,
) -> dict[str, float]:
    annual_path = source_dir / 'annual_by_standard_kg.csv'
    audit_path = source_dir / 'audit_summary.json'
    if not annual_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(
            'annual_by_standard_kg.csv and audit_summary.json are required '
            'for reconciliation.'
        )

    annual_reference = pd.read_csv(annual_path).set_index('standard')
    if '全部标准' not in annual_reference.index:
        raise ValueError('Annual reference is missing the 全部标准 total row.')
    standard_reference = annual_reference.drop(index='全部标准')
    grouped = all_months.groupby('standard')[RECONCILE_COLUMNS].sum()
    if set(grouped.index) != set(standard_reference.index):
        raise ValueError(
            'Monthly and annual summaries have different emission-standard labels.'
        )

    for column in RECONCILE_COLUMNS:
        observed = grouped[column].reindex(standard_reference.index).to_numpy(float)
        expected = standard_reference[column].to_numpy(float)
        if not np.allclose(observed, expected, rtol=1e-12, atol=1e-5):
            raise ValueError(
                f'Monthly aggregation does not reconcile to annual_by_standard_kg.csv '
                f'for {column}.'
            )
        if not np.isclose(
            float(grouped[column].sum()),
            float(annual_reference.loc['全部标准', column]),
            rtol=1e-12,
            atol=1e-5,
        ):
            raise ValueError(
                f'Standard totals do not reconcile to the annual total for {column}.'
            )

    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('status') != 'completed_summary_and_checks':
        raise ValueError('Audit summary does not report completed checks.')
    if audit.get('all_conservation_passed') is not True:
        raise ValueError('Audit summary reports failed conservation checks.')
    for column in ['CO2_kg', 'distance_km']:
        if not np.isclose(
            float(all_months[column].sum()),
            float(audit['annual_kg'][column]),
            rtol=1e-12,
            atol=1e-5,
        ):
            raise ValueError(f'Monthly data does not reconcile to audit for {column}.')
    return {
        'CO2_kg': float(all_months.CO2_kg.sum()),
        'distance_km': float(all_months.distance_km.sum()),
        'CO2_speed_added_kg': float(all_months.CO2_speed_added_kg.sum()),
        'distance_added_km': float(all_months.distance_added_km.sum()),
    }


def main(source_dir: Path, out: Path) -> None:
    source_dir = source_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [
        source_dir / f'{MONTHLY_PREFIX}{month}.csv' for month in MONTHS
    ] + [
        source_dir / 'annual_by_standard_kg.csv',
        source_dir / 'audit_summary.json',
    ]
    missing = [path.name for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Missing required source files in {source_dir}: {", ".join(missing)}'
        )
    source_signatures = [signature(path) for path in source_paths]
    all_months, monthly_files = read_monthly_files(source_dir)
    reconciled_totals = check_annual_reconciliation(all_months, source_dir)
    audit = json.loads(
        (source_dir / 'audit_summary.json').read_text(encoding='utf-8')
    )
    monthly_audits = audit['monthly_audits']
    input_points = sum(int(item['input_points']) for item in monthly_audits)
    excluded_points = sum(int(item['excluded_points']) for item in monthly_audits)

    annual = (
        all_months.groupby('enterprise_id', as_index=False)[['CO2_kg', 'distance_km']]
        .sum()
        .rename(columns={
            'distance_km': 'observed_vkt_2024_km',
            'CO2_kg': 'calibrated_tailpipe_co2_2024_kg',
        })
    )
    if annual.empty or annual.enterprise_id.duplicated().any():
        raise ValueError('Annual enterprise aggregation is empty or has duplicate IDs.')
    if not annual.observed_vkt_2024_km.gt(0).all():
        raise ValueError('Aggregated enterprise VKT must be positive.')
    if not annual.calibrated_tailpipe_co2_2024_kg.gt(0).all():
        raise ValueError('Aggregated enterprise CO2 must be positive.')
    annual['demand_year'] = 2024
    annual['emission_coverage_factor_already_applied'] = 1.0
    input_path = out / 'steel_enterprise_inputs_2024.csv'
    write_csv(input_path, annual)

    analyze_enterprises(input_path, out)
    if source_signatures != [signature(path) for path in source_paths]:
        raise AssertionError('A source file changed during post-processing.')

    report = {
        'status': 'PASS',
        'baseline_year': 2024,
        'monthly_files': [path.name for path in monthly_files],
        'enterprise_count': int(annual.enterprise_id.nunique()),
        'monthly_source_row_count': int(len(all_months)),
        'aggregation': (
            'Sum CO2_kg and distance_km by enterprise_id across all 12 months, '
            'emission standards and categories.'
        ),
        'supplement_columns_handling': (
            'CO2_speed_added_kg and distance_added_km are audit subsets already '
            'included in CO2_kg and distance_km; they are not added a second time.'
        ),
        'reconciled_totals': reconciled_totals,
        'source_data_quality': {
            'input_points': input_points,
            'excluded_points': excluded_points,
            'excluded_share_pct': 100 * excluded_points / input_points,
            'excluded_point_emissions_assumed_zero': False,
        },
        'checks': {
            'all_12_months_present': True,
            'no_duplicate_monthly_enterprise_standard_category_keys': True,
            'monthly_standard_totals_match_annual_reference': True,
            'monthly_totals_match_audit_summary': True,
            'original_source_files_unchanged': True,
        },
        'source_signatures': source_signatures,
    }
    (out / 'steel_enterprise_postprocess_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    print(json.dumps({
        'status': 'PASS',
        'output': str(out),
        'enterprise_count': report['enterprise_count'],
        'annual_CO2_t': reconciled_totals['CO2_kg'] / 1000,
        'annual_vkt_km': reconciled_totals['distance_km'],
        'postprocess_validation': str(
            out / 'steel_enterprise_postprocess_validation.json'
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args.source_dir, args.output_dir)
