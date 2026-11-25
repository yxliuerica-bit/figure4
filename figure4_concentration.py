"""Plot plant coverage, CO2 abatement and within-group electrification depth.

2024 demand is fixed. Plants are ordered by annual observed vehicle-km, with
plant_id as a tie-break. CO2 is the existing x3 calibrated tailpipe baseline.
No cost, trip-count, upstream-electricity or annual-rollout model is used.

Usage in Figure4: python figure4_concentration.py
Optional: --input-dir DIR --output-dir DIR
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

HERE = Path(__file__).resolve().parent
PREFIX = 'Figure4_concentration'
STEM = 'Figure4_E_plant_coverage_depth_2024'
STEM_SHARE = 'Figure4_E_plant_coverage_share_depth_2024'
STEM_EN = 'Figure4_E_plant_coverage_depth_2024_English'
GOALS = [5,10,20,30,40,50,60,70,80,90,95,100]
PLOT_GOALS = [30,50,80]


def signature(path):
    path = Path(path)
    s = path.stat()
    return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def write_csv(path, frame):
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')


def build(plants, expected_coverage_factor=3.0):
    required = ['plant_id','observed_vkt_2024_km','calibrated_tailpipe_co2_2024_kg',
                'demand_year','emission_coverage_factor_already_applied']
    assert set(required).issubset(plants.columns)
    assert plants[required].notna().all().all()
    assert plants.plant_id.is_unique
    assert plants.demand_year.eq(2024).all()
    factors = plants.emission_coverage_factor_already_applied.to_numpy(dtype=float)
    assert np.isfinite(factors).all() and (factors > 0).all()
    if expected_coverage_factor is not None:
        assert plants.emission_coverage_factor_already_applied.eq(
            expected_coverage_factor).all()
    assert plants.observed_vkt_2024_km.gt(0).all()
    assert plants.calibrated_tailpipe_co2_2024_kg.gt(0).all()
    d = plants.sort_values(['observed_vkt_2024_km','plant_id'],
                          ascending=[False,True],kind='stable').reset_index(drop=True)
    n = len(d)
    d.insert(0,'transport_demand_rank',np.arange(1,n+1))
    d['plant_coverage_pct'] = 100*d.transport_demand_rank/n
    d['annual_calibrated_tailpipe_CO2_t'] = d.calibrated_tailpipe_co2_2024_kg/1000
    d['cumulative_observed_vkt_km'] = d.observed_vkt_2024_km.cumsum()
    d['cumulative_vkt_share_pct'] = 100*d.cumulative_observed_vkt_km/d.observed_vkt_2024_km.sum()
    d['cumulative_calibrated_tailpipe_CO2_t'] = d.annual_calibrated_tailpipe_CO2_t.cumsum()
    total = float(d.annual_calibrated_tailpipe_CO2_t.sum())
    d['full_group_CO2_reduction_pct'] = 100*d.cumulative_calibrated_tailpipe_CO2_t/total
    # Set only mathematically exact endpoints; retain every observed intermediate value.
    d.loc[n-1,'full_group_CO2_reduction_pct'] = 100.
    d.loc[n-1,'cumulative_vkt_share_pct'] = 100.
    capacity = d.full_group_CO2_reduction_pct.to_numpy()
    goals=[]
    for target in GOALS:
        k=int(np.searchsorted(capacity,target,side='left'))+1
        r=d.iloc[k-1]
        q=target/r.full_group_CO2_reduction_pct
        assert k==1 or capacity[k-2]<target
        assert capacity[k-1]>=target and 0<q<=1+1e-12
        goals.append({'target_national_CO2_reduction_pct':target,
            'annual_CO2_avoided_t':target/100*total,
            'prefix_plants_required':k,'prefix_plant_share_pct':100*k/n,
            'full_prefix_CO2_reduction_pct':float(r.full_group_CO2_reduction_pct),
            'prefix_common_electrification_depth_pct':float(q*100),
            'prefix_full_vkt_share_pct':float(r.cumulative_vkt_share_pct),
            'prefix_electrified_national_vkt_share_pct':float(q*r.cumulative_vkt_share_pct),
            'uniform_plants_required':n,'uniform_common_electrification_depth_pct':target,
            'uniform_electrified_national_vkt_share_pct':target})
    goals=pd.DataFrame(goals)
    surfaces=[]
    for q in [0,10,20,40,60,80,100]:
        s=d[['transport_demand_rank','plant_coverage_pct',
             'full_group_CO2_reduction_pct','cumulative_vkt_share_pct']].copy()
        s['common_within_group_electrification_pct']=q
        s['national_CO2_reduction_pct']=s.full_group_CO2_reduction_pct*q/100
        s['annual_CO2_avoided_t']=s.national_CO2_reduction_pct/100*total
        s['national_electrified_vkt_share_pct']=s.cumulative_vkt_share_pct*q/100
        surfaces.append(s)
    return d,goals,pd.concat(surfaces,ignore_index=True)


def plot(ranked,goals,out,share_axis=False,english=False,entity_cn='电厂',
         entity_en='Power plants',figure_stem=None,share_figure_stem=None,
         english_figure_stem=None):
    font_family='Arial' if english else 'Microsoft YaHei'
    plt.rcParams.update({'font.family':font_family,'font.size':6.5,
        'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'path','mathtext.fontset':'dejavusans'})
    n=len(ranked)
    k=ranked.transport_demand_rank.to_numpy(float)
    if share_axis:
        x=ranked.plant_coverage_pct.to_numpy(float)
        x_end=100.
        xlim=(float(x[0]),100.)
        if n == 1:
            xlim=(80.,100.)
        xticks=[.1,.2,.5,1,2,5,10,20,50,100]
        xlabel=f'参与转型的{entity_cn}比例（%）'
        xformatter=FuncFormatter(lambda v,pos:f'{v:g}')
        stem=share_figure_stem or STEM_SHARE
    else:
        x=k
        x_end=float(n)
        xlim=(1,n)
        if n == 1:
            xlim=(.8,1.25)
        xticks=[1,5,10,20,50,100,200,500,n]
        xlabel=(f'{entity_en} adopting clean freight' if english
                else f'清洁货运转型{entity_cn}数量')
        xformatter=FuncFormatter(lambda v,pos:f'{int(v):,}')
        stem=(english_figure_stem or STEM_EN) if english else (figure_stem or STEM)
    capacity=ranked.full_group_CO2_reduction_pct.to_numpy(float)
    figsize=(80/25.4,58.4/25.4) if english else (89/25.4,65/25.4)
    fig=plt.figure(figsize=figsize)
    axes_rect=[.14,.14,.622,.705] if english else [.115,.14,.665,.705]
    colorbar_rect=[.815,.14,.023,.705] if english else [.835,.14,.023,.705]
    ax=fig.add_axes(axes_rect)
    # With the ranking fixed, common depth q is uniquely determined by (K,target).
    y=np.linspace(0,100,1001)
    X,Y=np.meshgrid(x,y)
    required=100*Y/capacity[None,:]
    required=np.ma.masked_where(required>100+1e-10,required)
    levels=[0,20,40,60,80,100]
    cmap=plt.get_cmap('YlGn',5)
    norm=BoundaryNorm(levels,cmap.N)
    # Each isoband has analytic boundaries q*capacity(K). Fill these directly
    # to avoid white masked-grid wedges near small integer K on a log axis.
    for lower,upper in zip(levels[:-1],levels[1:]):
        ax.fill_between(x,capacity*lower/100,capacity*upper/100,
                        color=cmap(norm((lower+upper)/2)),zorder=0,lw=0)
    fill=ScalarMappable(norm=norm,cmap=cmap)
    ax.fill_between(x,capacity,100,color='#ffffff',zorder=1)
    if n > 1:
        ax.contour(X,Y,required,levels=[20,40,60,80],
                            colors=['#566d58','#566d58','#ffffff','#ffffff'],
                            linewidths=.5,zorder=2)
    orange,blue='#d55f32','#176d9a'
    ax.plot(x,capacity,color=orange,lw=1.3,zorder=4)
    ax.plot([x_end,x_end],[0,100],color=blue,lw=1.4,zorder=4,clip_on=False)
    # Integer-prefix markers meet each goal, sometimes at depth just below 100%.
    for target in PLOT_GOALS:
        r=goals.loc[goals.target_national_CO2_reduction_pct.eq(target)].iloc[0]
        x_goal=float(r.prefix_plant_share_pct if share_axis else r.prefix_plants_required)
        # Full-width reference lines, not only the segment between strategies.
        ax.axhline(target,color='#6f7c82',lw=.7,ls=(0,(4,3)),zorder=5)
        ax.vlines(x_goal,0,100,color=orange,lw=.7,ls=(0,(4,3)),zorder=5)
        ax.scatter([x_goal],[target],s=22,marker='D',fc='white',ec=orange,lw=1,zorder=6)
        ax.scatter([x_end],[target],s=22,marker='o',fc='white',ec=blue,lw=1,zorder=6,clip_on=False)
    ax.scatter([x_end],[100],s=24,fc='white',ec='#35424a',lw=1,zorder=7,clip_on=False)
    ax.set_xscale('log')
    ax.set(xlim=xlim,ylim=(0,100),
           xlabel=xlabel,
           ylabel=(r'CO$_2$ reduction (%)' if english
                   else r'CO$_2$减排比例（%）'))
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(xformatter)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out',length=2.5,color='#66747c')
    cax=fig.add_axes(colorbar_rect)
    cb=fig.colorbar(fill,cax=cax,ticks=levels,spacing='proportional')
    cb.ax.set_yticklabels([f'{v}%' for v in levels])
    cb.ax.tick_params(labelsize=6.5,length=2)
    cb.outline.set_edgecolor('#c3ccd1')
    if english:
        colorbar_label=('Plant-level share of truck VKT electrified (%)'
                        if entity_cn == '电厂'
                        else f'Share of truck VKT electrified at {entity_en.lower()} (%)')
    else:
        colorbar_label=f'{entity_cn}运输里程电动化比例'
    cb.set_label(colorbar_label,fontsize=6.5,labelpad=6)
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'{stem}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)


def main(input_dir,out):
    out.mkdir(parents=True,exist_ok=True)
    source=input_dir/'Figure4_inputs_2024.csv'
    provenance=input_dir/'Figure4_input_provenance.json'
    sources=[source,provenance]
    signatures=[signature(p) for p in sources]
    plants=pd.read_csv(source,float_precision='round_trip')
    prov=json.loads(provenance.read_text(encoding='utf-8'))
    assert prov['baseline_year']==2024 and prov['demand_multiplier']==1.
    ranked,goals,surface=build(plants)
    # Verify the x3 emission snapshot against its annual baseline when available.
    annual_source=Path(prov['source_directory'])/'emission_by_plant_annual_2024_kg.csv'
    annual_check=None
    if annual_source.is_file():
        sources.append(annual_source)
        signatures.append(signature(annual_source))
        raw=pd.read_csv(annual_source,float_precision='round_trip').set_index('电厂ID')
        assert set(raw.index)==set(ranked.plant_id)
        np.testing.assert_allclose(raw.reindex(ranked.plant_id).CO2_kg,
            ranked.calibrated_tailpipe_co2_2024_kg,rtol=1e-12,atol=1e-5)
        annual_check=True
    n=len(ranked)
    cap=ranked.full_group_CO2_reduction_pct.to_numpy()
    assert (np.diff(cap)>0).all() and cap[-1]==100
    assert np.isclose(surface.loc[surface.common_within_group_electrification_pct.eq(100),
                                  'national_CO2_reduction_pct'],cap,rtol=1e-12).all()
    uniform=surface.loc[surface.transport_demand_rank.eq(n)]
    np.testing.assert_allclose(uniform.national_CO2_reduction_pct,
                               uniform.common_within_group_electrification_pct,rtol=1e-12)
    # Independent prefix sums and target-depth reconstruction.
    e=ranked.calibrated_tailpipe_co2_2024_kg.to_numpy()
    for _,g in goals.iterrows():
        k=int(g.prefix_plants_required)
        capacity=100*e[:k].sum()/e.sum()
        assert np.isclose(capacity,g.full_prefix_CO2_reduction_pct,rtol=1e-12)
        assert np.isclose(capacity*g.prefix_common_electrification_depth_pct/100,
                          g.target_national_CO2_reduction_pct,rtol=1e-12)
    write_csv(out/f'{PREFIX}_ranked_plants_2024.csv',ranked)
    write_csv(out/f'{PREFIX}_targets_2024.csv',goals)
    write_csv(out/f'{PREFIX}_surface_2024.csv',surface)
    plot(ranked,goals,out)
    plot(ranked,goals,out,share_axis=True)
    plot(ranked,goals,out,english=True)
    assert signatures==[signature(p) for p in sources]
    r=ranked.iloc[31]
    validation={'status':'PASS','baseline_year':2024,'target_year':2035,'demand_multiplier':1.,
        'plot_style':'three single-column logarithmic-x plots: Chinese plant-count and plant-share axes, plus English plant-count axis; axes and colorbar only',
        'full_width_horizontal_reference_lines_pct':PLOT_GOALS,
        'plant_count':n,'baseline_observed_vkt_km':float(ranked.observed_vkt_2024_km.sum()),
        'baseline_calibrated_tailpipe_CO2_t':float(ranked.annual_calibrated_tailpipe_CO2_t.sum()),
        'ranking':'annual observed vehicle-km descending; plant_id ascending tie break',
        'surface_definition':'R(K,q)=q*sum(E_i,i<=K)/sum(E_i), q in [0,1]',
        'within_group_assumption':'each selected plant electrifies the same representative share q of its transport vehicle-km; within-plant CO2 reduction proportional to q',
        'concentrated_boundary':'q=100%; all selected prefix plants fully electrified',
        'uniform_path':'K=N; common q increases from 0 to 100%',
        'targets':'minimum integer K in the fixed VKT ranking; target marker may require q slightly below 100%; not a globally optimal subset',
        'top32':{'plant_share_pct':float(r.plant_coverage_pct),
                 'vkt_share_pct':float(r.cumulative_vkt_share_pct),
                 'CO2_share_pct':float(r.full_group_CO2_reduction_pct)},
        'checks':{'annual_baseline_snapshot_matches':annual_check,
                  'ranking_and_unique_plant_ids':True,'monotone_positive_prefix_CO2':True,
                  'uniform_path_reduction_equals_common_depth':True,
                  'targets_minimum_integer_prefix_and_depth_reconstruction':True,
                  'full_scale_both_paths_converge':True,'no_extra_x3_emission_scaling':True,
                  'input_files_unchanged':True},
        'source_signatures':signatures,
        'limitations':['Avoided CO2 is tailpipe only; electricity-generation emissions are not included.',
            'All routes assumed technically electrifiable; no charging, grid, vehicle-range, cost or adoption constraints.',
            'Plant counts measure implementation coverage, not equal investment or administrative costs.',
            'Electricity demand and freight activity fixed at 2024; no annual rollout process.',
            'K is integer. Contours/curves connect evaluations between adjacent integer prefixes solely for display.',
            'Below the boundary, common-q scenarios distribute partial electrification equally across selected plants. They are not the original greedy rule with all earlier plants complete and only the last plant partial.']}
    (out/f'{PREFIX}_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    text='''Figure4最终版：电厂覆盖范围—减排目标—厂内转型深度（对数坐标）

需求固定为2024年；排放使用已经乘3的覆盖校准尾气CO2，不能再次乘3。
输入为Figure4_inputs_2024.csv。按各厂年度运输车辆公里降序、plant_id升序打破并列。
不使用成本、车次表、唯一车辆数、发电端排放或年度推进过程。

K为参与转型的排名前K家电厂，q为所选各厂内部共同的运输车辆公里电动化比例。
在每厂按代表性运输活动替代的假设下，厂内尾气CO2减排比例等于q。
R(K,q)=q*前K家基准CO2之和/全国基准CO2。
输出三张对数横轴图：中文电厂数量横轴图、中文电厂比例横轴图（100K/N，%）及英文电厂数量横轴图；比例图采用适配比例范围的刻度。
中文图尺寸为89×65mm，英文数量横轴图为80×58.4mm；图中文字均为6.5pt。
图中仅保留坐标轴及色标文字；不绘制标题、图例、说明框、引线说明、脚注或等值线文字。
在全国减排30%、50%、80%处绘制贯穿全图的水平虚线，保留相应的策略交点标记。
纵轴为全国尾气CO2减排比例。色面表示不同K、q下的全国减排比例；橙色边界以上的白色区域表示q>100%，该前缀电厂组合无法达到目标。
橙色边界为q=100%的集中转型，蓝色竖线为K=N的全国统一转型。
两者不是同资金或同运输工作量比较，电厂数也不能直接换算为组织成本。

整数K：目标表寻找既定排序下满足目标的最小整数K，再计算共同q。
目标标记可能略低于全转型边界。最小K仅针对该排序，并非所有电厂组合的全局最优解。
色面下方情景是所选每厂相同q，不是此前前厂全转、最后一家部分转型的逐厂贪心配置。
集中边界的整数K处与前K家全转型完全一致；点间连线和等值线仅为可视化插值。
所有运输均假定可电动化，未加入车辆续航、补能、电网、接受度或预算约束。

复现：在Figure4目录运行 python figure4_concentration.py
可选：--input-dir <已有Figure4目录> --output-dir <输出目录>
输出含中文数量横轴图、中文比例横轴图及英文数量横轴图（各含PNG/PDF/SVG）、逐厂排序表、目标表、若干共同q的情景表及核查JSON。
仅保留本最终版绘图脚本和PNG/PDF/SVG，以及输入、来源和核查结果。
成本参数保留作来源记录，本最终图不调用成本模型。旧试验图、脚本及试算结果已清理。
基准输入、原成本工作簿及Figure4外的原始排放数据不修改。
'''
    (out/'README_Figure4.txt').write_text(text,encoding='utf-8')
    config={'demand_year':2024,'transition_target_year':2035,'demand_multiplier':1.,
        'model':'plant_prefix_coverage_and_common_electrification_depth',
        'ranking':'annual observed vehicle-km descending; plant_id ascending tie break',
        'xscale':'log','cost_included':False,'horizontal_reference_lines_pct':PLOT_GOALS,
        'figure_stem':STEM,'additional_figure_stem':STEM_SHARE,
        'english_figure_stem':STEM_EN,
        'x_axes':['plant_count','plant_share_pct'],
        'entry_script':'figure4_concentration.py'}
    (out/'figure4_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','output':str(out),'top32':validation['top32'],
        'plot_goals':goals.loc[goals.target_national_CO2_reduction_pct.isin(PLOT_GOALS),
            ['target_national_CO2_reduction_pct','prefix_plants_required',
             'prefix_plant_share_pct','prefix_common_electrification_depth_pct']].to_dict('records')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=HERE)
    p.add_argument('--output-dir',type=Path,default=HERE)
    args=p.parse_args()
    main(args.input_dir,args.output_dir)
