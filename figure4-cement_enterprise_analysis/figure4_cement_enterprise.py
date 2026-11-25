"""Plot 2024 cement enterprise freight electrification from the annual CSV."""

import argparse
from pathlib import Path

from figure4_steel_enterprise import main as analyze_enterprises

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / 'cement_enterprise_analysis'
INPUT_NAME = 'cement_enterprise_inputs_2024.csv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_OUTPUT_DIR / INPUT_NAME)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    analyze_enterprises(
        args.input,
        args.output_dir,
        industry='Cement',
        entity_cn='水泥企业',
        entity_en='Cement enterprises',
        process='cement production',
    )
