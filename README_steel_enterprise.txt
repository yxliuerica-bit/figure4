钢铁企业货运电动化覆盖—减排分析

本脚本沿用 Figure4 的分析逻辑，将分析单元从电厂替换为钢铁企业。适用对象是企业货运车辆的活动量及尾气排放，不是炼钢工艺排放。当前目录没有钢铁企业专属数据；请准备并核验数据后再运行，不应直接把电厂输入当作钢铁企业结果。

直接提供输入 CSV 时，每家企业一行，必须包含以下列：
- enterprise_id：企业唯一标识，按文本读取。
- observed_vkt_2024_km：2024 年货运车辆公里，必须大于 0。
- calibrated_tailpipe_co2_2024_kg：与上述货运活动对应的 2024 年尾气 CO2 基准排放，kg，必须大于 0。
- demand_year：全部为 2024。
- emission_coverage_factor_already_applied：排放估算中已应用的覆盖系数，必须为正数；仅作记录，脚本不会再次乘该系数。

当前已有的钢铁月度数据可由后处理入口直接汇总。它读取 summary_2024 下的 12 个 `.enterprise_standard_202401.csv` 至 `.enterprise_standard_202412.csv`，按企业合计 `CO2_kg` 和 `distance_km`，并与 `annual_by_standard_kg.csv` 及 `audit_summary.json` 对账。`CO2_speed_added_kg` 和 `distance_added_km` 是已包含在主总量中的核验子项，不能重复相加。原目录只读，不会修改源文件。

来源审计记录有部分缺失距离轨迹点已被排除；其未知里程和排放不能视作零。逐企业结果代表现有可用记录的汇总，应结合源目录中的核验说明理解覆盖边界。

从钢铁月度明细运行：
python figure4_steel_enterprise_from_monthly.py
python figure4_steel_enterprise_from_monthly.py --source-dir <summary_2024目录> --output-dir <输出目录>

后处理会在输出目录写出逐企业年度输入 `steel_enterprise_inputs_2024.csv`、分析表、图片，以及后处理和模型核查 JSON。

直接提供整理好的企业年度 CSV 时运行：
python figure4_steel_enterprise.py --input steel_enterprise_inputs_2024.csv
python figure4_steel_enterprise.py --input <CSV路径> --output-dir <输出目录>

排序按年度货运 VKT 降序、enterprise_id 升序打破并列。K 是排名前 K 家企业，q 是所选企业内部相同的货运 VKT 电动化比例；全国尾气减排率为 q × 前 K 家企业的基准尾气 CO2 占比。目标表给出达到 5% 至 100% 减排目标的最小整数 K 及对应共同电动化深度。此排序不是所有企业组合的全局最优解。

输出包含企业排序表、减排目标表、K-q 情景面表、核查 JSON，以及中文企业数量横轴图、中文企业比例横轴图和英文数量横轴图（PNG/PDF/SVG）。所有图以 2024 年活动量和尾气排放为基准。

解释限制：结果只代表货运车辆尾气 CO2，不包含钢铁生产过程排放或发电端排放；假定货运活动可电动化且减排与活动替代比例线性相关，不含续航、充电、电网、成本或采用约束。企业数代表覆盖数量，不代表投资或实施成本。
