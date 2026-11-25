"""Analyze freight electrification coverage and CO2 reduction for steel enterprises.

The input must contain one row per steel enterprise with observed 2024 freight
vehicle-kilometres and calibrated 2024 freight tailpipe CO2 emissions.

Usage: python figure4_steel_enterprise.py --input steel_enterprise_inputs_2024.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from figure4_concentration import (
    PLOT_GOALS,
    build,
    plot,
    signature,
    write_csv,
)

HERE = Path(__file__).resolve().parent
PREFIX = 'Steel_enterprise_concentration'
FIGURE_STEM = 'Steel_enterprise_coverage_depth_2024'
SHARE_FIGURE_STEM = 'Steel_enterprise_coverage_share_depth_2024'
ENGLISH_FIGURE_STEM = 'Steel_enterprise_coverage_depth_2024_English'
REQUIRED_COLUMNS = [
    'enterprise_id',
    'observed_vkt_2024_km',
    'calibrated_tailpipe_co2_2024_kg',
    'demand_year',
    'emission_coverage_factor_already_applied',
]


def load_input(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={'enterprise_id': 'string'},
                       float_precision='round_trip')
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f'Input is missing required columns: {", ".join(missing)}')
    if data.empty:
        raise ValueError('Input must contain at least one steel enterprise.')
    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError('Required input fields must not contain blank values.')
    data['enterprise_id'] = data.enterprise_id.str.strip()
    if data.enterprise_id.eq('').any() or not data.enterprise_id.is_unique:
        raise ValueError('enterprise_id values must be non-empty and unique.')
    if not data.demand_year.eq(2024).all():
        raise ValueError('All rows must describe baseline year 2024.')
    for column in [
        'observed_vkt_2024_km',
        'calibrated_tailpipe_co2_2024_kg',
        'emission_coverage_factor_already_applied',
    ]:
        data[column] = pd.to_numeric(data[column], errors='raise')
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f'{column} must contain finite numeric values.')
    if not data.observed_vkt_2024_km.gt(0).all():
        raise ValueError('observed_vkt_2024_km must be greater than zero.')
    if not data.calibrated_tailpipe_co2_2024_kg.gt(0).all():
        raise ValueError('calibrated_tailpipe_co2_2024_kg must be greater than zero.')
    if not data.emission_coverage_factor_already_applied.gt(0).all():
        raise ValueError('Emission coverage factors must be greater than zero.')
    return data


def main(input_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    input_signature = signature(input_path)
    enterprises = load_input(input_path)

    model_input = enterprises.rename(columns={'enterprise_id': 'plant_id'})
    ranked, goals, surface = build(model_input, expected_coverage_factor=None)

    plot(
        ranked,
        goals,
        out,
        entity_cn='钢铁企业',
        entity_en='Steel enterprises',
        figure_stem=FIGURE_STEM,
        share_figure_stem=SHARE_FIGURE_STEM,
        english_figure_stem=ENGLISH_FIGURE_STEM,
    )
    plot(
        ranked,
        goals,
        out,
        share_axis=True,
        entity_cn='钢铁企业',
        entity_en='Steel enterprises',
        figure_stem=FIGURE_STEM,
        share_figure_stem=SHARE_FIGURE_STEM,
        english_figure_stem=ENGLISH_FIGURE_STEM,
    )
    plot(
        ranked,
        goals,
        out,
        english=True,
        entity_cn='钢铁企业',
        entity_en='Steel enterprises',
        figure_stem=FIGURE_STEM,
        share_figure_stem=SHARE_FIGURE_STEM,
        english_figure_stem=ENGLISH_FIGURE_STEM,
    )

    ranked_output = ranked.rename(columns={
        'transport_demand_rank': 'enterprise_rank',
        'plant_id': 'enterprise_id',
        'plant_coverage_pct': 'enterprise_coverage_pct',
    })
    goals_output = goals.rename(columns={
        'prefix_plants_required': 'prefix_enterprises_required',
        'prefix_plant_share_pct': 'prefix_enterprise_share_pct',
        'uniform_plants_required': 'uniform_enterprises_required',
    })
    surface_output = surface.rename(columns={
        'transport_demand_rank': 'enterprise_rank',
        'plant_coverage_pct': 'enterprise_coverage_pct',
    })

    write_csv(out / f'{PREFIX}_ranked_enterprises_2024.csv', ranked_output)
    write_csv(out / f'{PREFIX}_targets_2024.csv', goals_output)
    write_csv(out / f'{PREFIX}_surface_2024.csv', surface_output)

    total_emissions_t = float(ranked.annual_calibrated_tailpipe_CO2_t.sum())
    checks = {
        'ranking_and_unique_enterprise_ids': True,
        'positive_monotone_prefix_CO2': bool(
            (np.diff(ranked.full_group_CO2_reduction_pct.to_numpy()) > 0).all()),
        'full_scale_paths_converge': bool(np.isclose(
            ranked.full_group_CO2_reduction_pct.iloc[-1], 100.0)),
        'input_file_unchanged': input_signature == signature(input_path),
    }
    if not all(checks.values()):
        raise AssertionError(f'Validation failed: {checks}')

    validation = {
        'status': 'PASS',
        'baseline_year': 2024,
        'entity': 'steel enterprise',
        'enterprise_count': len(ranked),
        'baseline_observed_vkt_km': float(ranked.observed_vkt_2024_km.sum()),
        'baseline_calibrated_freight_tailpipe_CO2_t': total_emissions_t,
        'ranking': 'annual observed freight vehicle-km descending; enterprise_id ascending tie break',
        'surface_definition': 'R(K,q)=q*sum(E_i,i<=K)/sum(E_i), q in [0,1]',
        'within_group_assumption': (
            'each selected enterprise electrifies the same representative share q '
            'of its freight vehicle-km; freight tailpipe CO2 reduction is proportional to q'
        ),
        'targets': (
            'minimum integer K in the fixed VKT ranking; this is not a globally '
            'optimal enterprise subset'
        ),
        'coverage_factor_handling': (
            'The input factor is retained as metadata; emissions are not scaled again.'
        ),
        'source_signature': input_signature,
        'checks': checks,
        'limitations': [
            'This analyzes freight-vehicle tailpipe emissions, not steelmaking process emissions.',
            'Avoided CO2 excludes electricity-generation emissions.',
            'All freight activity is assumed technically electrifiable; vehicle, charging, grid, cost and adoption constraints are omitted.',
            'Enterprise counts measure implementation coverage, not equal investment or administrative cost.',
            'Freight activity is fixed at 2024; no annual rollout process is modeled.',
        ],
    }
    (out / f'{PREFIX}_validation.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    print(json.dumps({
        'status': 'PASS',
        'output': str(out),
        'enterprise_count': len(ranked),
        'baseline_freight_tailpipe_CO2_t': total_emissions_t,
        'target_summary': goals_output.loc[
            goals_output.target_national_CO2_reduction_pct.isin(PLOT_GOALS),
            ['target_national_CO2_reduction_pct',
             'prefix_enterprises_required',
             'prefix_enterprise_share_pct',
             'prefix_common_electrification_depth_pct'],
        ].to_dict('records'),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        type=Path,
        default=HERE / 'steel_enterprise_inputs_2024.csv',
        help='CSV with one row per steel enterprise (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=HERE / 'steel_enterprise_analysis',
        help='Directory for tables, plots and validation JSON',
    )
    args = parser.parse_args()
    main(args.input, args.output_dir)
