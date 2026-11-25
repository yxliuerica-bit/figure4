"""Aggregate 12 monthly cement emission fact Parquets into a 2024 enterprise CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'cement_enterprise_analysis'
INPUT_NAME = 'cement_enterprise_inputs_2024.csv'
# The main totals are used directly. Repair diagnostics such as
# distance_added_km and CO2_speed_added_kg are not additional activity.
MEASURES = ['distance_km', 'CO2_kg']
CEMENT_INDUSTRIES = ('shuini', 'cement', '水泥')


def resolve_column(schema, requested, candidates, path):
    if requested is not None and requested in schema.names:
        return requested
    if requested is None and candidates[0] in schema.names:
        return candidates[0]
    matches = [name for name in candidates if name in schema.names]
    if requested is None and len(matches) == 1:
        return matches[0]
    option = '--enterprise-col' if candidates[0] == 'enterprise_id' else '--distance-col'
    raise ValueError(f'{path.name}: specify {option}; available columns: {schema.names}')


def aggregate_month(path, enterprise_col, distance_col, batch_size):
    month = path.stem.removeprefix('emission_fact_')
    parquet = pq.ParquetFile(path)
    try:
        schema = parquet.schema_arrow
        enterprise = resolve_column(schema, enterprise_col, ['enterprise_id'], path)
        distance = resolve_column(
            schema, distance_col, ['distance_km', 'vkt_km', 'vehicle_km'], path,
        )
        columns = [enterprise, distance, 'CO2_kg']
        if len(set(columns)) != 3 or 'CO2_kg' not in schema.names:
            raise ValueError(f'{path.name}: expected distinct enterprise, distance and CO2_kg columns.')

        metadata = {
            name: allowed for name, allowed in
            [('industry', CEMENT_INDUSTRIES), ('month', (month,))]
            if name in schema.names
        }
        if set(columns).intersection(metadata):
            raise ValueError(f'{path.name}: industry and month are metadata columns.')
        columns.extend(metadata)

        totals = {}
        rows = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pandas(types_mapper=pd.ArrowDtype)
            for name, allowed in metadata.items():
                values = data[name].astype('string').str.strip()
                if name == 'industry':
                    values = values.str.lower()
                invalid = ~values.isin(allowed)
                if invalid.any():
                    examples = values[invalid].drop_duplicates().head(5).tolist()
                    raise ValueError(
                        f'{path.name}: unexpected {name} after row {rows}: {examples}; '
                        f'expected {allowed}.'
                    )
            data = data.rename(columns={enterprise: 'enterprise_id', distance: 'distance_km'})
            ids = data.enterprise_id.astype('string').str.strip()
            if ids.isna().any() or ids.eq('').any():
                raise ValueError(f'{path.name}: missing enterprise ID in batch after row {rows}.')
            data['enterprise_id'] = ids
            for column in MEASURES:
                values = pd.to_numeric(data[column], errors='coerce').to_numpy(
                    dtype=float, na_value=np.nan,
                )
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f'{path.name}: invalid {column} in batch after row {rows}.')
                data[column] = values

            grouped = data.groupby('enterprise_id', sort=False)[MEASURES].sum()
            for key, km, co2 in grouped.itertuples():
                previous_km, previous_co2 = totals.get(key, (0.0, 0.0))
                totals[key] = (previous_km + km, previous_co2 + co2)
            rows += len(data)
    finally:
        parquet.close()

    result = pd.DataFrame.from_dict(totals, orient='index', columns=MEASURES)
    result.index.name = 'enterprise_id'
    co2_total = float(result.CO2_kg.sum())
    manifest_path = path.with_suffix('.manifest.json')
    audit = {
        'month': month,
        'parquet': path.name,
        'rows': rows,
        'enterprise_column': enterprise,
        'distance_column': distance,
        'distance_km': float(result.distance_km.sum()),
        'CO2_kg': co2_total,
        'parquet_metadata_columns_checked': list(metadata),
        'manifest_checked': manifest_path.is_file(),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        if (str(manifest['month']) != month or str(manifest['industry']).strip().lower() not in CEMENT_INDUSTRIES
                or manifest['rows'] != rows
                or not np.isclose(manifest['totals_kg']['CO2_kg'], co2_total, rtol=1e-7, atol=1e-6)):
            raise ValueError(f'{path.name}: month, industry, row count or CO2 total disagrees with manifest.')
        for field in ('excluded_points', 'missing_policy', 'in_multiplier', 'out_multiplier'):
            if field in manifest:
                audit[field] = manifest[field]
    return result, audit


def main(args):
    paths = [args.source_dir / f'emission_fact_2024{month:02d}.parquet'
             for month in range(1, 13)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing files in {args.source_dir}: {", ".join(missing)}')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive.')

    monthly = []
    audits = []
    for path in paths:
        result, audit = aggregate_month(path, args.enterprise_col, args.distance_col, args.batch_size)
        monthly.append(result)
        audits.append(audit)
        print(f'{path.name}: {audit["rows"]:,} rows, {len(result):,} enterprises', flush=True)

    annual = pd.concat(monthly).groupby(level=0)[MEASURES].sum()
    if annual.empty or not np.isfinite(annual.to_numpy()).all():
        raise ValueError('Annual totals must contain finite data for at least one enterprise.')
    zero_both = annual[MEASURES].eq(0).all(axis=1)
    excluded_enterprises = int(zero_both.sum())
    annual = annual.loc[~zero_both].copy()
    invalid = annual.index[annual[MEASURES].le(0).any(axis=1)]
    if len(invalid) or annual.empty:
        raise ValueError(
            'Plotting requires positive annual distance and CO2 for each active enterprise. '
            f'Enterprises with only one positive measure (first 10): {invalid[:10].tolist()}'
        )
    annual = annual.rename(columns={
        'distance_km': 'observed_vkt_2024_km',
        'CO2_kg': 'calibrated_tailpipe_co2_2024_kg',
    })
    annual['demand_year'] = 2024
    annual['emission_coverage_factor_already_applied'] = 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / INPUT_NAME
    annual.to_csv(output, encoding='utf-8-sig')
    audit_path = args.output_dir / 'cement_enterprise_aggregation_validation.json'
    audit_path.write_text(json.dumps({
        'status': 'PASS',
        'baseline_year': 2024,
        'enterprise_count': len(annual),
        'excluded_zero_distance_and_zero_co2_enterprises': excluded_enterprises,
        'annual_distance_km': float(annual.observed_vkt_2024_km.sum()),
        'annual_CO2_kg': float(annual.calibrated_tailpipe_co2_2024_kg.sum()),
        'aggregation_multiplier': 1.0,
        'monthly': audits,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Wrote {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--enterprise-col', help='Enterprise ID column (default: enterprise_id)')
    parser.add_argument('--distance-col', help='Total vehicle-kilometres column (default: distance_km)')
    parser.add_argument('--batch-size', type=int, default=250_000)
    main(parser.parse_args())
