#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岚图汽车遥测数据 — 清洗、插值、分析、可视化、导出与报告生成
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = ROOT_DIR / "plots"

TRIP_DETAIL_CSV = ROOT_DIR / "行程分析明细.csv"
CHARGE_DETAIL_CSV = ROOT_DIR / "充电分析明细.csv"
DETAIL_EXCEL = ROOT_DIR / "分析明细汇总.xlsx"
REPORT_TXT = ROOT_DIR / "数据挖掘分析报告.txt"

DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}

TIMESTAMP_COLUMN_CANDIDATES = (
    "时间戳",
    "时间轴",
    "timestamp",
    "Timestamp",
    "TIME",
    "时间",
    "采集时间",
    "记录时间",
)

DISCRETE_COLUMNS = (
    "档位",
    "Ready",
    "驾驶模式",
    "混动模式",
    "能量回收模式",
    "智能驾驶是否开启",
    "充电枪状态",
    "充电状态",
    "前空调开关",
    "后空调开关",
    "进风循环模式请求",
    "遮阳帘开关",
    "方向盘加热状态",
    "主驾座椅加热状态",
    "副驾座椅加热状态",
    "二排左座椅加热状态",
    "二排右座椅加热状态",
    "三排左座椅加热状态",
    "三排右座椅加热状态",
)

TRIP_START_MOVING_SECONDS = 10
TRIP_END_IDLE_SECONDS = 300
CHARGING_STATUS_ACTIVE = 5
LOW_SPEED_MAX = 60.0
MAX_PLOT_POINTS = 50_000
ML_SAMPLE_SIZE = 150_000
ML_RANDOM_STATE = 42


@dataclass
class AnalysisContext:
    """汇总全流程指标，供控制台、报告与导出复用。"""

    source_file: str = ""
    generated_at: str = ""
    rows: int = 0
    cols: int = 0
    missing_before: int = 0
    missing_after: int = 0
    continuous_cols: int = 0
    discrete_cols: int = 0
    time_column: str = ""
    overview_missing: dict[str, int] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    trip_summary: dict[str, Any] = field(default_factory=dict)
    charge_summary: dict[str, Any] = field(default_factory=dict)
    motor_summary: dict[str, Any] = field(default_factory=dict)
    ml_summary: dict[str, Any] = field(default_factory=dict)
    plot_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def setup_chinese_font() -> None:
    preferred = (
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    for font in font_manager.fontManager.ttflist:
        path_lower = (font.fname or "").lower()
        if any(k in path_lower for k in ("yahei", "simhei", "pingfang", "notosanscjk", "sourcehan")):
            plt.rcParams["font.sans-serif"] = [font.name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def find_data_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name == Path(__file__).name:
            continue
        if path.suffix.lower() in DATA_EXTENSIONS:
            files.append(path)
    return files


def read_data_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(file_path, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法识别 CSV 编码: {file_path}")
    if suffix in (".xlsx", ".xls"):
        engine = "openpyxl" if suffix == ".xlsx" else None
        return pd.read_excel(file_path, engine=engine)
    raise ValueError(f"不支持的文件格式: {file_path}")


def find_timestamp_column(columns: pd.Index) -> str | None:
    col_set = {str(c).strip() for c in columns}
    for candidate in TIMESTAMP_COLUMN_CANDIDATES:
        if candidate in col_set:
            return candidate
    lower_map = {str(c).strip().lower(): str(c).strip() for c in columns}
    for candidate in TIMESTAMP_COLUMN_CANDIDATES:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _existing_columns(df: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    return [c for c in names if c in df.columns]


def _format_duration(seconds: float) -> str:
    if pd.isna(seconds):
        return "—"
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds / 3600:.2f} 小时"


def _label_segments(
    active: np.ndarray,
    ready: np.ndarray | None = None,
    min_active_len: int = 1,
    end_idle_seconds: int = TRIP_END_IDLE_SECONDS,
    end_on_ready_zero: bool = False,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    n = len(active)
    in_seg = False
    start = 0
    run_count = 0
    gap_count = 0

    for i in range(n):
        if not in_seg:
            if active[i]:
                run_count += 1
            else:
                run_count = 0
            if run_count >= min_active_len:
                in_seg = True
                start = i - min_active_len + 1
                gap_count = 0
        else:
            end_now = False
            if end_on_ready_zero and ready is not None and ready[i] == 0:
                end_now = True
            if not active[i]:
                gap_count += 1
                if gap_count >= end_idle_seconds:
                    end_now = True
            else:
                gap_count = 0
            if end_now:
                end_idx = i - gap_count if gap_count > 0 else i
                segments.append((start, max(start, end_idx)))
                in_seg = False
                run_count = 0
                gap_count = 0

    if in_seg:
        segments.append((start, n - 1))
    return segments


def _section(title: str, width: int = 70) -> list[str]:
    line = "═" * width
    return ["", line, f"  {title}", line]


def _bullet(lines: list[str], indent: str = "  ") -> list[str]:
    return [f"{indent}• {line}" for line in lines]


# ---------------------------------------------------------------------------
# 数据清洗与插值
# ---------------------------------------------------------------------------


def capture_overview(df: pd.DataFrame, file_path: Path, ctx: AnalysisContext) -> None:
    ctx.source_file = file_path.name
    ctx.rows = len(df)
    ctx.cols = len(df.columns)
    missing = df.isna().sum()
    ctx.overview_missing = {
        str(col): int(count) for col, count in missing[missing > 0].items()
    }


def print_data_overview(df: pd.DataFrame, file_path: Path, ctx: AnalysisContext) -> None:
    capture_overview(df, file_path, ctx)
    print("=" * 60)
    print(f"数据文件: {file_path.name}")
    print(f"完整路径: {file_path}")
    print("-" * 60)
    print(f"行数: {ctx.rows:,}  |  列数: {ctx.cols}")
    print("-" * 60)
    print("各列数据类型:")
    print(df.dtypes.to_string())
    print("-" * 60)
    if not ctx.overview_missing:
        print("缺失值: 无")
    else:
        print("缺失值统计（仅显示存在缺失的列）:")
        for col, count in sorted(ctx.overview_missing.items(), key=lambda x: -x[1]):
            pct = count / ctx.rows * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    print("=" * 60)


def convert_timestamp_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna()
        if not sample.empty and sample.max() > 1e12:
            df[column] = pd.to_datetime(series, unit="ms", errors="coerce")
        else:
            df[column] = pd.to_datetime(series, unit="s", errors="coerce")
    else:
        df[column] = pd.to_datetime(series, errors="coerce")

    invalid = df[column].isna().sum() - series.isna().sum()
    if invalid > 0:
        print(f"警告: 「{column}」有 {invalid:,} 条记录无法解析为日期时间，已设为 NaT。")

    print(f"「{column}」已转换为 datetime64，示例:")
    print(df[column].dropna().head(3).to_string(index=False))
    return df


def fill_missing_values(df: pd.DataFrame, time_col: str, ctx: AnalysisContext) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    discrete = _existing_columns(df, DISCRETE_COLUMNS)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    continuous = [c for c in numeric_cols if c not in discrete and c != "vin"]

    ctx.continuous_cols = len(continuous)
    ctx.discrete_cols = len(discrete)
    ctx.missing_before = int(df.isna().sum().sum())

    if continuous:
        df[continuous] = (
            df[continuous]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    if discrete:
        df[discrete] = df[discrete].ffill().bfill()

    ctx.missing_after = int(df.isna().sum().sum())
    print("\n【插值补齐】")
    print(f"  连续数值列 ({ctx.continuous_cols} 列): 线性插值 + 前向/后向填充")
    print(f"  离散状态列 ({ctx.discrete_cols} 列): 前向填充 + 后向填充")
    if ctx.missing_after == 0:
        print(f"  确认: 缺失值已由 {ctx.missing_before:,} 条清零，当前数据集无缺失值。")
    else:
        print(f"  已处理 {ctx.missing_before:,} 条缺失值，剩余 {ctx.missing_after:,} 条。")
    return df


# ---------------------------------------------------------------------------
# 统计分析 & 可视化
# ---------------------------------------------------------------------------


def compute_statistics(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if "车速" in df.columns:
        stats["平均车速_km_h"] = float(df["车速"].mean())
    if "最低电芯温度" in df.columns:
        stats["最低电芯温度_°C"] = float(df["最低电芯温度"].min())
    if "最高电芯温度" in df.columns:
        stats["最高电芯温度_°C"] = float(df["最高电芯温度"].max())
    front_col, rear_col = "前电机机械功率", "后电机功率"
    if front_col in df.columns and rear_col in df.columns:
        drive = df[front_col].fillna(0) + df[rear_col].fillna(0)
        stats["平均驱动电机功率_kW"] = float(drive.mean())
    return stats


def print_statistics(df: pd.DataFrame, ctx: AnalysisContext) -> None:
    ctx.stats = compute_statistics(df)
    print("\n【统计分析】")
    if "平均车速_km_h" in ctx.stats:
        print(f"  平均车速: {ctx.stats['平均车速_km_h']:.2f} km/h")
    if "最低电芯温度_°C" in ctx.stats and "最高电芯温度_°C" in ctx.stats:
        print(f"  最低电芯温度: {ctx.stats['最低电芯温度_°C']:.2f} °C")
        print(f"  最高电芯温度: {ctx.stats['最高电芯温度_°C']:.2f} °C")
    if "平均驱动电机功率_kW" in ctx.stats:
        print(f"  平均驱动电机功率: {ctx.stats['平均驱动电机功率_kW']:.2f} kW（前电机 + 后电机）")


def _prepare_plot_dataframe(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    plot_df = df.sort_values(time_col).dropna(subset=[time_col]).copy()
    if len(plot_df) <= MAX_PLOT_POINTS:
        return plot_df
    plot_df = plot_df.set_index(time_col)
    rule = "10s" if len(plot_df) < 500_000 else "30s"
    plot_df = plot_df.resample(rule).mean(numeric_only=True).dropna(how="all").reset_index()
    print(f"  绘图数据已按 {rule} 重采样至 {len(plot_df):,} 个点。")
    return plot_df


def create_visualizations(df: pd.DataFrame, time_col: str, ctx: AnalysisContext) -> None:
    setup_chinese_font()
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_df = _prepare_plot_dataframe(df, time_col)
    t = plot_df[time_col]
    ctx.plot_files = []

    print("\n【数据可视化】")
    print(f"  图表保存目录: {PLOTS_DIR}")

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    if "车速" in plot_df.columns:
        ax1.plot(t, plot_df["车速"], label="车速 (km/h)", color="#1f77b4", linewidth=0.8)
    ax1.set_ylabel("车速 (km/h)")
    ax1.set_xlabel("时间")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    if "加速踏板开度" in plot_df.columns:
        ax2.plot(t, plot_df["加速踏板开度"], label="加速踏板开度", color="#ff7f0e", linewidth=0.6, alpha=0.85)
    if "制动踏板状态（开度）" in plot_df.columns:
        ax2.plot(t, plot_df["制动踏板状态（开度）"], label="制动踏板开度", color="#d62728", linewidth=0.6, alpha=0.85)
    ax2.set_ylabel("踏板开度 (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("车速与加速/制动踏板开度随时间变化")
    fig1.autofmt_xdate()
    fig1.tight_layout()
    p1 = PLOTS_DIR / "01_车速与踏板开度.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    ctx.plot_files.append(str(p1))
    print(f"  已保存: {p1.name}")

    if "SOC" in plot_df.columns:
        fig2, ax = plt.subplots(figsize=(14, 5))
        ax.plot(t, plot_df["SOC"], color="#2ca02c", linewidth=0.8)
        ax.set_title("电池 SOC 随时间变化")
        ax.set_xlabel("时间")
        ax.set_ylabel("SOC (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        fig2.autofmt_xdate()
        fig2.tight_layout()
        p2 = PLOTS_DIR / "02_SOC趋势.png"
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        ctx.plot_files.append(str(p2))
        print(f"  已保存: {p2.name}")

    temp_cols = _existing_columns(
        plot_df,
        ("前电机温度", "后电机温度", "发电机温度", "最低电芯温度", "最高电芯温度"),
    )
    if temp_cols:
        fig3, ax = plt.subplots(figsize=(14, 6))
        colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]
        for i, col in enumerate(temp_cols):
            ax.plot(t, plot_df[col], label=col, color=colors[i % len(colors)], linewidth=0.8)
        ax.set_title("前后电机、发电机与电芯温度对比")
        ax.set_xlabel("时间")
        ax.set_ylabel("温度 (°C)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig3.autofmt_xdate()
        fig3.tight_layout()
        p3 = PLOTS_DIR / "03_温度对比.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        ctx.plot_files.append(str(p3))
        print(f"  已保存: {p3.name}")


# ---------------------------------------------------------------------------
# 行程 / 充电 / 电机效率 / 机器学习
# ---------------------------------------------------------------------------


def analyze_trips(df: pd.DataFrame, time_col: str, ctx: AnalysisContext) -> pd.DataFrame:
    print("\n【行程分析】")
    if "车速" not in df.columns:
        print("  跳过: 缺少车速列。")
        return pd.DataFrame()

    speed = df["车速"].fillna(0).to_numpy()
    ready = df["Ready"].fillna(1).to_numpy() if "Ready" in df.columns else np.ones(len(df))
    segments = _label_segments(
        speed > 0,
        ready=ready,
        min_active_len=TRIP_START_MOVING_SECONDS,
        end_on_ready_zero=True,
    )

    if not segments:
        print("  未识别到有效行程。")
        return pd.DataFrame()

    times = df[time_col]
    records = []
    for idx, (start_i, end_i) in enumerate(segments, start=1):
        seg = df.iloc[start_i : end_i + 1]
        duration_sec = (times.iloc[end_i] - times.iloc[start_i]).total_seconds()
        if duration_sec <= 0:
            continue

        mileage = np.nan
        if "总里程" in seg.columns:
            odo = seg["总里程"].dropna()
            if len(odo) >= 2:
                mileage = float(odo.iloc[-1] - odo.iloc[0])
                if mileage < 0:
                    mileage = abs(mileage)

        soc_start = soc_end = soc_used = np.nan
        if "SOC" in seg.columns:
            soc_vals = seg["SOC"].dropna()
            if len(soc_vals) >= 1:
                soc_start = float(soc_vals.iloc[0])
                soc_end = float(soc_vals.iloc[-1])
                soc_used = max(0.0, soc_start - soc_end)

        records.append(
            {
                "行程编号": idx,
                "开始时间": times.iloc[start_i],
                "结束时间": times.iloc[end_i],
                "持续时间_秒": duration_sec,
                "持续时间": _format_duration(duration_sec),
                "里程_km": mileage,
                "起始SOC_%": soc_start,
                "结束SOC_%": soc_end,
                "消耗SOC_%": soc_used,
            }
        )

    trip_df = pd.DataFrame(records)
    if trip_df.empty:
        print("  未识别到有效行程。")
        return trip_df

    longest = trip_df.loc[trip_df["持续时间_秒"].idxmax()]
    ctx.trip_summary = {
        "总行程数": len(trip_df),
        "规则": (
            f"车速>0 持续 {TRIP_START_MOVING_SECONDS}s 开始；"
            f"车速=0 持续 {TRIP_END_IDLE_SECONDS // 60} 分钟或 Ready=0 结束"
        ),
        "平均持续时间_秒": float(trip_df["持续时间_秒"].mean()),
        "总里程_km": float(trip_df["里程_km"].sum(skipna=True)),
        "总消耗SOC_%": float(trip_df["消耗SOC_%"].sum(skipna=True)),
        "最长行程": {
            "行程编号": int(longest["行程编号"]),
            "开始时间": str(longest["开始时间"]),
            "结束时间": str(longest["结束时间"]),
            "持续时间": _format_duration(longest["持续时间_秒"]),
            "持续时间_秒": float(longest["持续时间_秒"]),
            "里程_km": float(longest["里程_km"]) if pd.notna(longest["里程_km"]) else None,
            "起始SOC_%": float(longest["起始SOC_%"]) if pd.notna(longest["起始SOC_%"]) else None,
            "结束SOC_%": float(longest["结束SOC_%"]) if pd.notna(longest["结束SOC_%"]) else None,
            "消耗SOC_%": float(longest["消耗SOC_%"]) if pd.notna(longest["消耗SOC_%"]) else None,
        },
    }

    print(f"  总行程数: {ctx.trip_summary['总行程数']}")
    print(f"  规则: {ctx.trip_summary['规则']}")
    print("-" * 60)
    show = trip_df[["行程编号", "持续时间", "里程_km", "消耗SOC_%"]].head(30)
    print(show.to_string(index=False, float_format="%.2f"))
    if len(trip_df) > 30:
        print(f"  ... 其余 {len(trip_df) - 30} 条行程已省略，详见导出文件。")
    print("-" * 60)
    lt = ctx.trip_summary["最长行程"]
    print("  【最长行程】")
    print(f"    行程编号: {lt['行程编号']}")
    print(f"    时间范围: {lt['开始时间']} ~ {lt['结束时间']}")
    print(f"    持续时间: {lt['持续时间']}")
    if lt.get("里程_km") is not None:
        print(f"    里程: {lt['里程_km']:.2f} km")
    if lt.get("消耗SOC_%") is not None:
        print(
            f"    SOC: {lt['起始SOC_%']:.2f}% → {lt['结束SOC_%']:.2f}%"
            f"（消耗 {lt['消耗SOC_%']:.2f}%）"
        )
    print(
        f"  汇总 — 平均持续时间: {_format_duration(ctx.trip_summary['平均持续时间_秒'])}，"
        f"总里程(可计): {ctx.trip_summary['总里程_km']:.2f} km，"
        f"总消耗SOC: {ctx.trip_summary['总消耗SOC_%']:.2f}%"
    )
    return trip_df


def analyze_charging(df: pd.DataFrame, time_col: str, ctx: AnalysisContext) -> pd.DataFrame:
    print("\n【充电分析】")
    has_status = "充电状态" in df.columns
    has_power = "充电功率" in df.columns
    if not has_status and not has_power:
        print("  跳过: 缺少充电状态/充电功率列。")
        return pd.DataFrame()

    charging = np.zeros(len(df), dtype=bool)
    if has_status:
        charging |= df["充电状态"].fillna(0).to_numpy() == CHARGING_STATUS_ACTIVE
    if has_power:
        charging |= df["充电功率"].fillna(0).to_numpy() > 0

    segments = _label_segments(charging, min_active_len=30, end_idle_seconds=60)
    if not segments:
        print("  未识别到充电片段。")
        return pd.DataFrame()

    times = df[time_col]
    records = []
    for idx, (start_i, end_i) in enumerate(segments, start=1):
        seg = df.iloc[start_i : end_i + 1]
        duration_sec = (times.iloc[end_i] - times.iloc[start_i]).total_seconds()
        if duration_sec <= 0:
            continue

        soc_start = soc_end = np.nan
        if "SOC" in seg.columns:
            soc_vals = seg["SOC"].dropna()
            if len(soc_vals):
                soc_start = float(soc_vals.iloc[0])
                soc_end = float(soc_vals.iloc[-1])

        avg_power = np.nan
        if "充电功率" in seg.columns:
            p = seg["充电功率"].replace(0, np.nan).dropna()
            if len(p):
                avg_power = float(p.mean())

        records.append(
            {
                "充电次数": idx,
                "开始时间": times.iloc[start_i],
                "结束时间": times.iloc[end_i],
                "充电耗时_秒": duration_sec,
                "充电耗时": _format_duration(duration_sec),
                "初始SOC_%": soc_start,
                "结束SOC_%": soc_end,
                "SOC增量_%": (soc_end - soc_start) if pd.notna(soc_start) and pd.notna(soc_end) else np.nan,
                "平均充电功率_kW": avg_power,
            }
        )

    charge_df = pd.DataFrame(records)
    ctx.charge_summary = {
        "充电次数": len(charge_df),
        "规则": f"充电状态={CHARGING_STATUS_ACTIVE}(在充) 或 充电功率>0，连续≥30秒",
        "平均充电功率_kW": float(charge_df["平均充电功率_kW"].mean(skipna=True))
        if charge_df["平均充电功率_kW"].notna().any()
        else None,
        "总充电时长_秒": float(charge_df["充电耗时_秒"].sum()),
    }

    print(f"  充电次数: {ctx.charge_summary['充电次数']}")
    print(f"  识别规则: {ctx.charge_summary['规则']}")
    print("-" * 60)
    show = charge_df[
        ["充电次数", "充电耗时", "初始SOC_%", "结束SOC_%", "平均充电功率_kW"]
    ].head(20)
    print(show.to_string(index=False, float_format="%.2f"))
    if len(charge_df) > 20:
        print(f"  ... 其余 {len(charge_df) - 20} 次充电已省略，详见导出文件。")
    if ctx.charge_summary["平均充电功率_kW"] is not None:
        print(
            f"  汇总 — 平均充电功率: {ctx.charge_summary['平均充电功率_kW']:.2f} kW，"
            f"总充电时长: {_format_duration(ctx.charge_summary['总充电时长_秒'])}"
        )
    return charge_df


def _motor_rpm(df: pd.DataFrame) -> pd.Series:
    rpm = pd.Series(0.0, index=df.index)
    if "发电机转速" in df.columns:
        rpm = df["发电机转速"].fillna(0).clip(lower=0)
    if "发动机转速" in df.columns:
        rpm = np.maximum(rpm, df["发动机转速"].fillna(0).clip(lower=0))
    return rpm


def _calc_efficiency(power_kw: pd.Series, voltage: pd.Series, current: pd.Series) -> pd.Series:
    electric_kw = (voltage * current) / 1000.0
    return (power_kw / electric_kw.replace(0, np.nan)).clip(lower=0, upper=1.5)


def analyze_motor_efficiency(df: pd.DataFrame, ctx: AnalysisContext) -> None:
    print("\n【电机效率分析】")
    rpm = _motor_rpm(df)
    speed = df["车速"].fillna(0) if "车速" in df.columns else pd.Series(0, index=df.index)

    configs = [
        ("前电机", "前电机机械功率", "前动电机电压", "前电机电流"),
        ("后电机", "后电机功率", "后电机电压", "后电机电流"),
    ]

    motors: dict[str, Any] = {}
    all_eff_rows: list[pd.DataFrame] = []

    for name, p_col, v_col, i_col in configs:
        if not all(c in df.columns for c in (p_col, v_col, i_col)):
            print(f"  {name}: 缺少功率/电压/电流列，跳过。")
            continue

        mask = (df[p_col] > 0) & (df[i_col] > 0) & (df[v_col] > 0) & (rpm > 0)
        eff = _calc_efficiency(df.loc[mask, p_col], df.loc[mask, v_col], df.loc[mask, i_col])
        valid = eff.notna() & np.isfinite(eff)
        if valid.sum() == 0:
            print(f"  {name}: 无有效行驶样本。")
            continue

        sub = pd.DataFrame({"电机": name, "车速": speed.loc[mask].values[valid], "效率": eff.values[valid]})
        all_eff_rows.append(sub)
        motors[name] = {
            "样本数": int(valid.sum()),
            "平均效率": float(sub["效率"].mean()),
            "低速平均效率": float(sub[sub["车速"] <= LOW_SPEED_MAX]["效率"].mean()),
            "高速平均效率": float(sub[sub["车速"] > LOW_SPEED_MAX]["效率"].mean()),
        }
        print(f"  {name} 有效样本: {motors[name]['样本数']:,}，平均效率: {motors[name]['平均效率']:.4f}")

    if not all_eff_rows:
        return

    combined = pd.concat(all_eff_rows, ignore_index=True)
    low = combined[combined["车速"] <= LOW_SPEED_MAX]
    high = combined[combined["车速"] > LOW_SPEED_MAX]

    ctx.motor_summary = {
        "各电机": motors,
        "综合平均效率": float(combined["效率"].mean()),
        "低速": {"样本数": len(low), "平均效率": float(low["效率"].mean())},
        "高速": {"样本数": len(high), "平均效率": float(high["效率"].mean())},
        "速度分段阈值_km_h": LOW_SPEED_MAX,
    }

    print(f"  综合平均效率（前后电机合并）: {ctx.motor_summary['综合平均效率']:.4f}")
    print("  分速度段平均效率（综合）:")
    print(
        f"    低速 (≤{LOW_SPEED_MAX:.0f} km/h): 样本 {ctx.motor_summary['低速']['样本数']:,}，"
        f"平均效率 {ctx.motor_summary['低速']['平均效率']:.4f}"
    )
    print(
        f"    高速 (>{LOW_SPEED_MAX:.0f} km/h): 样本 {ctx.motor_summary['高速']['样本数']:,}，"
        f"平均效率 {ctx.motor_summary['高速']['平均效率']:.4f}"
    )
    for name, m in motors.items():
        print(f"    {name} — 低速: {m['低速平均效率']:.4f}，高速: {m['高速平均效率']:.4f}")


def _sample_for_ml(df: pd.DataFrame, n: int = ML_SAMPLE_SIZE) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=ML_RANDOM_STATE)


def train_predict_models(df: pd.DataFrame, ctx: AnalysisContext) -> None:
    print("\n【能耗与温度预测（随机森林）】")
    sample = _sample_for_ml(df)
    ctx.ml_summary["训练采样条数"] = len(sample)
    ctx.ml_summary["全量条数"] = len(df)
    print(f"  训练采样: {len(sample):,} 条（全量 {len(df):,} 条）")

    feat_a = _existing_columns(sample, ("车速", "加速踏板开度", "环境温度"))
    if "前电机机械功率" in sample.columns and "后电机功率" in sample.columns and len(feat_a) == 3:
        data_a = sample[feat_a + ["前电机机械功率", "后电机功率"]].copy()
        data_a["驱动电机总功率"] = data_a["前电机机械功率"] + data_a["后电机功率"]
        data_a = data_a[(data_a["驱动电机总功率"] >= 0) & data_a[feat_a].notna().all(axis=1)]
        if len(data_a) > 1000:
            X_a, y_a = data_a[feat_a].values, data_a["驱动电机总功率"].values
            X_tr, X_te, y_tr, y_te = train_test_split(X_a, y_a, test_size=0.2, random_state=ML_RANDOM_STATE)
            model_a = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=ML_RANDOM_STATE)
            model_a.fit(X_tr, y_tr)
            pred_a = model_a.predict(X_te)
            ctx.ml_summary["任务A"] = {
                "描述": "车速、加速踏板开度、环境温度 → 前/后电机总机械功率",
                "R2": float(r2_score(y_te, pred_a)),
                "MAE": float(mean_absolute_error(y_te, pred_a)),
                "特征重要性": {k: float(v) for k, v in zip(feat_a, model_a.feature_importances_)},
            }
            print(f"  任务A — 特征: {ctx.ml_summary['任务A']['描述']}")
            print(f"    R²  = {ctx.ml_summary['任务A']['R2']:.4f}")
            print(f"    MAE = {ctx.ml_summary['任务A']['MAE']:.4f}")
            imp = sorted(ctx.ml_summary["任务A"]["特征重要性"].items(), key=lambda x: -x[1])
            print("    特征重要性:", ", ".join(f"{k}: {v:.3f}" for k, v in imp))
        else:
            print("  任务A: 有效样本不足，跳过。")
    else:
        print("  任务A: 缺少必要列，跳过。")

    if "最高电芯温度" not in sample.columns or "环境温度" not in sample.columns:
        print("  任务B: 缺少必要列，跳过。")
        return

    data_b = sample.copy()
    front_p = data_b["前电机机械功率"].fillna(0) if "前电机机械功率" in data_b.columns else 0
    rear_p = data_b["后电机功率"].fillna(0) if "后电机功率" in data_b.columns else 0
    data_b["电机总功率"] = front_p + rear_p
    data_b["电机转速"] = _motor_rpm(data_b)
    feat_b = ["电机总功率", "电机转速", "环境温度"]
    data_b = data_b[feat_b + ["最高电芯温度"]].dropna()
    data_b = data_b[(data_b["最高电芯温度"] > -40) & (data_b["最高电芯温度"] < 80)]

    if len(data_b) > 1000:
        X_b, y_b = data_b[feat_b].values, data_b["最高电芯温度"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_b, test_size=0.2, random_state=ML_RANDOM_STATE)
        model_b = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=ML_RANDOM_STATE)
        model_b.fit(X_tr, y_tr)
        pred_b = model_b.predict(X_te)
        ctx.ml_summary["任务B"] = {
            "描述": "电机总功率、电机转速、环境温度 → 最高电芯温度",
            "R2": float(r2_score(y_te, pred_b)),
            "MAE": float(mean_absolute_error(y_te, pred_b)),
            "特征重要性": {k: float(v) for k, v in zip(feat_b, model_b.feature_importances_)},
        }
        print(f"  任务B — 特征: {ctx.ml_summary['任务B']['描述']}")
        print(f"    R²  = {ctx.ml_summary['任务B']['R2']:.4f}")
        print(f"    MAE = {ctx.ml_summary['任务B']['MAE']:.4f}")
        imp = sorted(ctx.ml_summary["任务B"]["特征重要性"].items(), key=lambda x: -x[1])
        print("    特征重要性:", ", ".join(f"{k}: {v:.3f}" for k, v in imp))
    else:
        print("  任务B: 有效样本不足，跳过。")


# ---------------------------------------------------------------------------
# 数据导出 & 报告生成
# ---------------------------------------------------------------------------


def export_analysis_details(trip_df: pd.DataFrame, charge_df: pd.DataFrame) -> None:
    """导出行程/充电明细 CSV，并写入 Excel 多 Sheet。"""
    print("\n【数据导出】")

    if not trip_df.empty:
        trip_df.to_csv(TRIP_DETAIL_CSV, index=False, encoding="utf-8-sig")
        print(f"  已保存: {TRIP_DETAIL_CSV.name}（{len(trip_df):,} 条行程）")
    else:
        print("  行程明细为空，跳过 CSV 导出。")

    if not charge_df.empty:
        charge_df.to_csv(CHARGE_DETAIL_CSV, index=False, encoding="utf-8-sig")
        print(f"  已保存: {CHARGE_DETAIL_CSV.name}（{len(charge_df):,} 次充电）")
    else:
        print("  充电明细为空，跳过 CSV 导出。")

    if not trip_df.empty or not charge_df.empty:
        try:
            with pd.ExcelWriter(DETAIL_EXCEL, engine="openpyxl") as writer:
                if not trip_df.empty:
                    trip_df.to_excel(writer, sheet_name="行程分析", index=False)
                if not charge_df.empty:
                    charge_df.to_excel(writer, sheet_name="充电分析", index=False)
            print(f"  已保存: {DETAIL_EXCEL.name}（Excel 多 Sheet 汇总）")
        except ImportError:
            print("  提示: 安装 openpyxl 后可生成 Excel 汇总文件 (pip install openpyxl)。")
        except Exception as exc:
            print(f"  Excel 导出失败: {exc}")


def generate_analysis_report(ctx: AnalysisContext) -> None:
    """生成排版精美的文本分析报告。"""
    width = 72
    lines: list[str] = []
    border = "╔" + "═" * (width - 2) + "╗"
    inner_end = "╚" + "═" * (width - 2) + "╝"
    sep = "╠" + "═" * (width - 2) + "╣"

    def box_row(text: str) -> None:
        padding = width - 4 - len(text.encode("gbk", errors="ignore")) + len(text)
        # 简化：按字符长度近似（中文报告以 utf-8 写文件）
        pad = max(0, width - 4 - len(text))
        lines.append("║ " + text + " " * pad + " ║")

    lines.append(border)
    box_row("岚图汽车遥测数据 · 数据挖掘分析报告")
    box_row(f"生成时间: {ctx.generated_at}")
    box_row(f"数据源: {ctx.source_file}")
    lines.append(sep)

    lines.extend(_section("一、数据概览", width))
    lines.extend(
        _bullet(
            [
                f"记录规模: {ctx.rows:,} 行 × {ctx.cols} 列",
                f"时间列: {ctx.time_column or '—'}",
                f"清洗前缺失单元格: {ctx.missing_before:,}",
                f"插值后缺失单元格: {ctx.missing_after:,}",
                f"连续数值列插值: {ctx.continuous_cols} 列 | 离散状态列填充: {ctx.discrete_cols} 列",
            ]
        )
    )
    if ctx.overview_missing:
        top_miss = sorted(ctx.overview_missing.items(), key=lambda x: -x[1])[:5]
        lines.append("  主要缺失列（清洗前，Top 5）:")
        for col, cnt in top_miss:
            lines.append(f"    - {col}: {cnt:,} ({cnt / ctx.rows * 100:.2f}%)")

    lines.extend(_section("二、核心统计指标", width))
    stat_lines = []
    if "平均车速_km_h" in ctx.stats:
        stat_lines.append(f"平均车速: {ctx.stats['平均车速_km_h']:.2f} km/h")
    if "最低电芯温度_°C" in ctx.stats:
        stat_lines.append(f"全时段最低电芯温度: {ctx.stats['最低电芯温度_°C']:.2f} °C")
    if "最高电芯温度_°C" in ctx.stats:
        stat_lines.append(f"全时段最高电芯温度: {ctx.stats['最高电芯温度_°C']:.2f} °C")
    if "平均驱动电机功率_kW" in ctx.stats:
        stat_lines.append(f"平均驱动电机功率: {ctx.stats['平均驱动电机功率_kW']:.2f} kW")
    lines.extend(_bullet(stat_lines or ["（无可用统计项）"]))

    lines.extend(_section("三、行程分析", width))
    if ctx.trip_summary:
        ts = ctx.trip_summary
        lines.extend(
            _bullet(
                [
                    f"总行程数: {ts['总行程数']}",
                    f"切分规则: {ts['规则']}",
                    f"平均持续时间: {_format_duration(ts['平均持续时间_秒'])}",
                    f"累计里程(可计): {ts['总里程_km']:.2f} km",
                    f"累计消耗 SOC: {ts['总消耗SOC_%']:.2f}%",
                ]
            )
        )
        lt = ts.get("最长行程", {})
        if lt:
            lines.append("  最长行程:")
            lines.append(f"    - 编号 {lt.get('行程编号')} | {lt.get('开始时间')} ~ {lt.get('结束时间')}")
            lines.append(f"    - 时长 {lt.get('持续时间')}")
            if lt.get("里程_km") is not None:
                lines.append(f"    - 里程 {lt['里程_km']:.2f} km")
            if lt.get("消耗SOC_%") is not None:
                lines.append(
                    f"    - SOC {lt['起始SOC_%']:.2f}% → {lt['结束SOC_%']:.2f}%"
                    f"（消耗 {lt['消耗SOC_%']:.2f}%）"
                )
        lines.append(f"  明细文件: {TRIP_DETAIL_CSV.name}")
    else:
        lines.append("  （未识别到行程）")

    lines.extend(_section("四、充电分析", width))
    if ctx.charge_summary:
        cs = ctx.charge_summary
        lines.extend(
            _bullet(
                [
                    f"充电次数: {cs['充电次数']}",
                    f"识别规则: {cs['规则']}",
                    f"总充电时长: {_format_duration(cs['总充电时长_秒'])}",
                ]
            )
        )
        if cs.get("平均充电功率_kW") is not None:
            lines.append(f"  • 平均充电功率: {cs['平均充电功率_kW']:.2f} kW")
        lines.append(f"  明细文件: {CHARGE_DETAIL_CSV.name}")
    else:
        lines.append("  （未识别到充电片段）")

    lines.extend(_section("五、电机效率分析", width))
    if ctx.motor_summary:
        ms = ctx.motor_summary
        lines.append(f"  • 综合平均效率: {ms['综合平均效率']:.4f}")
        lines.append(
            f"  • 低速段 (≤{ms['速度分段阈值_km_h']:.0f} km/h): "
            f"样本 {ms['低速']['样本数']:,}，效率 {ms['低速']['平均效率']:.4f}"
        )
        lines.append(
            f"  • 高速段 (>{ms['速度分段阈值_km_h']:.0f} km/h): "
            f"样本 {ms['高速']['样本数']:,}，效率 {ms['高速']['平均效率']:.4f}"
        )
        for name, m in ms.get("各电机", {}).items():
            lines.append(
                f"  • {name}: 样本 {m['样本数']:,} | 平均 {m['平均效率']:.4f} | "
                f"低速 {m['低速平均效率']:.4f} | 高速 {m['高速平均效率']:.4f}"
            )
    else:
        lines.append("  （无有效效率样本）")

    lines.extend(_section("六、机器学习预测（随机森林）", width))
    lines.append(f"  • 训练采样: {ctx.ml_summary.get('训练采样条数', '—'):,} / 全量 {ctx.ml_summary.get('全量条数', '—'):,}")
    for task_key, label in (("任务A", "能耗预测"), ("任务B", "温度预测")):
        task = ctx.ml_summary.get(task_key)
        if not task:
            lines.append(f"  • {label}: 未执行或样本不足")
            continue
        lines.append(f"  • {label}: {task['描述']}")
        lines.append(f"      R²  = {task['R2']:.4f}")
        lines.append(f"      MAE = {task['MAE']:.4f}")
        imp = sorted(task["特征重要性"].items(), key=lambda x: -x[1])
        lines.append("      特征重要性: " + ", ".join(f"{k} {v:.3f}" for k, v in imp))

    lines.extend(_section("七、可视化产出", width))
    if ctx.plot_files:
        lines.extend(_bullet([Path(p).name for p in ctx.plot_files]))
        lines.append(f"  • 图表目录: {PLOTS_DIR}")
    else:
        lines.append("  （未生成图表）")

    lines.extend(_section("八、导出文件清单", width))
    exports = [
        TRIP_DETAIL_CSV.name,
        CHARGE_DETAIL_CSV.name,
        DETAIL_EXCEL.name,
        REPORT_TXT.name,
    ]
    lines.extend(_bullet(exports))
    lines.append("")
    lines.append(inner_end)

    report_body = "\n".join(lines)
    REPORT_TXT.write_text(report_body, encoding="utf-8")
    print(f"\n【分析报告】")
    print(f"  已保存: {REPORT_TXT.name}")


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------


def process_file(file_path: Path) -> pd.DataFrame:
    ctx = AnalysisContext(generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f"\n正在读取: {file_path.name} ...")
    df = read_data_file(file_path)
    print_data_overview(df, file_path, ctx)

    time_col = find_timestamp_column(df.columns)
    if time_col is None:
        print("未找到时间列，终止后续分析。")
        return df

    ctx.time_column = time_col
    print(f"\n识别到时间列: 「{time_col}」")
    df = convert_timestamp_column(df, time_col)
    df = fill_missing_values(df, time_col, ctx)

    print_statistics(df, ctx)
    create_visualizations(df, time_col, ctx)

    trip_df = analyze_trips(df, time_col, ctx)
    charge_df = analyze_charging(df, time_col, ctx)
    analyze_motor_efficiency(df, ctx)
    train_predict_models(df, ctx)

    export_analysis_details(trip_df, charge_df)
    generate_analysis_report(ctx)

    return df


def main() -> int:
    configure_stdout_utf8()

    data_files = find_data_files(ROOT_DIR)
    if not data_files:
        print(f"在 {ROOT_DIR} 下未找到 CSV 或 Excel 数据文件。")
        return 1

    print(f"共发现 {len(data_files)} 个数据文件:")
    for f in data_files:
        print(f"  - {f.name}")

    for file_path in data_files:
        process_file(file_path)

    print(
        "\n全部流程完成：数据清洗 → 插值补齐 → 统计可视化 → 深度分析 → "
        "明细导出 → 报告生成。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
