# -*- coding: utf-8 -*-
"""
Final-report chart source for the 30/4 EDA draft.

This script intentionally sits under `5 goc nhin` so the report charts are
traceable back to the source EDA notebooks:
- goc nhin 3: promotion_effectiveness.ipynb
- goc nhin 4: product_portfolio.ipynb
- goc nhin 6: seasonal_capital_misallocation.ipynb

Run from the repository/workspace root:
    python "The_4_Outliers_VinDatathon/5 góc nhìn/report_chart_source/make_report_charts.py"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data" / "clean"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

COMMITTED_STATUSES = ["paid", "delivered", "shipped", "returned"]
PEAK_MONTHS = [4, 5, 6]

sys.path.insert(0, str(PROJECT_ROOT))
try:
    from theme import COLORS, CATEGORY_COLORS, CLR_GRID, CLR_LABEL, CLR_MUTED, CLR_SPINE, CLR_TITLE
except Exception:  # pragma: no cover - fallback for standalone runs
    COLORS = {
        "blue": "#2563EB",
        "flame": "#EA580C",
        "emerald": "#16A34A",
        "red": "#DC2626",
        "violet": "#7C3AED",
        "cyan": "#0891B2",
    }
    CATEGORY_COLORS = {
        "Streetwear": "#2563EB",
        "Casual": "#EA580C",
        "GenZ": "#16A34A",
        "Outdoor": "#7C3AED",
    }
    CLR_GRID = "#E5E7EB"
    CLR_LABEL = "#374151"
    CLR_MUTED = "#6B7280"
    CLR_SPINE = "#D1D5DB"
    CLR_TITLE = "#111827"


@dataclass
class ChartMetrics:
    net_fulfilled_sales_b: float
    net_fulfilled_margin: float
    promo_margin: float
    nonpromo_margin: float
    margin_gap_pp: float
    promo_discount_burn_m: float
    offpeak_discount_burn_m: float
    offpeak_discount_share_fulfilled: float
    promo_existing_share_recent: float
    promo_new_share_recent: float
    corr_promo_net_revenue: float
    corr_promo_buyer_count: float
    revenue_day_ratio: float
    sessions_day_ratio: float
    peak_promo_rate: float
    offpeak_promo_rate: float
    streetwear_stockout_days: float
    outdoor_stockout_days: float
    top20_gp_share: float
    bottom30_gp_share: float
    negative_sku_count: int
    negative_sku_loss_m: float
    quarantine_sku_count: int
    quarantine_loss_m: float
    wrong_size_returns: int
    wrong_size_reason_share: float
    wrong_size_refund_m: float
    return_rate_order: float
    size_return_rate_spread_pp: float
    wrong_size_share_spread_pp: float


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def style_report_axes(ax, *, grid_axis: str = "x") -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, color=CLR_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=CLR_LABEL, labelsize=9.2)
    ax.xaxis.label.set_color(CLR_LABEL)
    ax.yaxis.label.set_color(CLR_LABEL)
    ax.title.set_color(CLR_TITLE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(CLR_SPINE)
    ax.spines["bottom"].set_color(CLR_SPINE)


def add_panel_label(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12.3, fontweight="semibold", pad=8)


def annotate_box(
    ax,
    text: str,
    x: float,
    y: float,
    *,
    color: str = COLORS["violet"],
    fontsize: float = 9.0,
    ha: str = "left",
    va: str = "top",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=color, linewidth=0.9),
    )


def load_clean_data() -> dict[str, pd.DataFrame]:
    names = [
        "customers",
        "inventory",
        "order_items",
        "orders",
        "products",
        "promotions",
        "returns",
        "sales",
        "web_traffic",
    ]
    data = {name: pd.read_parquet(DATA_DIR / f"{name}.parquet") for name in names}

    for frame_name, date_cols in {
        "orders": ["order_date"],
        "promotions": ["start_date", "end_date"],
        "returns": ["return_date"],
        "sales": ["date"],
        "web_traffic": ["date"],
        "inventory": ["snapshot_date"],
    }.items():
        for col in date_cols:
            data[frame_name][col] = pd.to_datetime(data[frame_name][col])
    return data


def build_line_items(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    order_items = data["order_items"].copy()
    orders = data["orders"].copy()
    products = data["products"].copy()

    f = (
        order_items.merge(orders, on="order_id", how="left")
        .merge(
            products[["product_id", "product_name", "category", "segment", "size", "cogs"]],
            on="product_id",
            how="left",
        )
    )
    f = f[f["order_status"].isin(COMMITTED_STATUSES)].copy()
    f["date"] = f["order_date"]
    f["year"] = f["date"].dt.year
    f["month"] = f["date"].dt.month
    f["is_peak"] = f["month"].isin(PEAK_MONTHS)
    f["discount_amount"] = f["discount_amount"].fillna(0.0)
    f["line_revenue"] = f["quantity"] * f["unit_price"]
    f["net_line"] = f["line_revenue"] - f["discount_amount"]
    f["cogs_line"] = f["quantity"] * f["cogs"]
    f["net_gp"] = f["net_line"] - f["cogs_line"]
    f["has_promo"] = (
        f["discount_amount"].gt(0)
        | f["promo_id"].fillna("").ne("No_Promo")
        | f["promo_id_2"].notna()
    )
    f["has_promo_k3"] = f["discount_amount"].gt(0)
    return f


def short_campaign_label(name: str) -> str:
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base, year = parts
        base = (
            base.replace(" Sale", "")
            .replace(" Launch", "")
            .replace(" Special", "")
            .replace(" Blowout", "")
        )
        vi_base = {
            "Year-End": "Cuối năm",
            "Fall": "Thu",
            "Mid-Year": "Giữa năm",
            "Spring": "Xuân",
            "Urban": "Urban",
            "Rural": "Rural",
        }.get(base, base)
        return f"{vi_base} '{year[-2:]}"
    return name[:18]


def recent_campaign_customer_mix(f: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    first_order = (
        f.groupby("customer_id", as_index=False)["date"]
        .min()
        .rename(columns={"date": "first_order_date"})
    )
    f_cust = f.merge(first_order, on="customer_id", how="left")
    top10 = promotions.sort_values("start_date", ascending=False).head(10)

    rows = []
    for _, promo in top10.iterrows():
        camp_items = f_cust[
            (
                f_cust["promo_id"].eq(promo["promo_id"])
                | f_cust["promo_id_2"].eq(promo["promo_id"])
            )
            & f_cust["has_promo"]
            & f_cust["date"].between(promo["start_date"], promo["end_date"])
        ].copy()
        if camp_items.empty:
            continue

        camp_items["is_new"] = camp_items["first_order_date"].ge(promo["start_date"])
        rows.append(
            {
                "promo_name": promo["promo_name"],
                "start_date": promo["start_date"],
                "net_new": camp_items.loc[camp_items["is_new"], "net_line"].sum(),
                "net_existing": camp_items.loc[~camp_items["is_new"], "net_line"].sum(),
            }
        )

    mix = pd.DataFrame(rows)
    mix["net_total"] = mix["net_existing"] + mix["net_new"]
    mix["existing_share"] = mix["net_existing"] / mix["net_total"]
    mix["new_share"] = mix["net_new"] / mix["net_total"]
    mix["short_label"] = mix["promo_name"].map(short_campaign_label)
    return mix


def compute_metrics(data: dict[str, pd.DataFrame]) -> tuple[ChartMetrics, pd.DataFrame, pd.DataFrame, pd.Series]:
    f = build_line_items(data)
    products = data["products"]
    returns = data["returns"]

    promo = f[f["has_promo"]]
    nonpromo = f[~f["has_promo"]]
    net_fulfilled_sales = f["net_line"].sum()
    net_fulfilled_gp = f["net_gp"].sum()
    promo_margin = promo["net_gp"].sum() / promo["net_line"].sum()
    nonpromo_margin = nonpromo["net_gp"].sum() / nonpromo["net_line"].sum()
    overall_margin = net_fulfilled_gp / net_fulfilled_sales

    daily = (
        f.groupby("date")
        .agg(
            net_rev=("net_line", "sum"),
            promo_rate=("has_promo", "mean"),
            unique_buyers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    campaign_mix = recent_campaign_customer_mix(f, data["promotions"])
    existing = campaign_mix["net_existing"].sum()
    new = campaign_mix["net_new"].sum()
    total = existing + new
    existing_share = existing / total
    new_share = new / total

    sales = data["sales"].copy()
    sales["year"] = sales["date"].dt.year
    sales["month"] = sales["date"].dt.month
    sales["is_peak"] = sales["month"].isin(PEAK_MONTHS)
    monthly_sales = sales.groupby(["year", "month", "is_peak"])["Revenue"].mean().reset_index()
    revenue_day_ratio = (
        monthly_sales.loc[monthly_sales["is_peak"], "Revenue"].mean()
        / monthly_sales.loc[~monthly_sales["is_peak"], "Revenue"].mean()
    )

    web = data["web_traffic"].copy()
    web["year"] = web["date"].dt.year
    web["month"] = web["date"].dt.month
    web["is_peak"] = web["month"].isin(PEAK_MONTHS)
    monthly_sessions = web.groupby(["year", "month", "is_peak"])["sessions"].mean().reset_index()
    sessions_day_ratio = (
        monthly_sessions.loc[monthly_sessions["is_peak"], "sessions"].mean()
        / monthly_sessions.loc[~monthly_sessions["is_peak"], "sessions"].mean()
    )

    monthly_promo = f.groupby(["year", "month", "is_peak"])["has_promo_k3"].mean().reset_index()
    peak_promo_rate = monthly_promo.loc[monthly_promo["is_peak"], "has_promo_k3"].mean()
    offpeak_promo_rate = monthly_promo.loc[~monthly_promo["is_peak"], "has_promo_k3"].mean()

    offpeak_discount = f.loc[~f["is_peak"], "discount_amount"].sum()
    promo_discount = f["discount_amount"].sum()

    cap_candidates = pd.read_csv(EDA_OUTPUT_DIR / "capital_reallocation_priority_candidates.csv")
    priority_stockouts = cap_candidates.set_index("category")["peak_stockout_days"]

    sku_gp = (
        f.groupby("product_id")["net_gp"]
        .sum()
        .reindex(products["product_id"].unique(), fill_value=0.0)
        .sort_values(ascending=False)
    )
    total_gp = sku_gp.sum()
    top_n = int(np.floor(len(sku_gp) * 0.20))
    bottom_n = int(np.ceil(len(sku_gp) * 0.30))
    top20_gp_share = sku_gp.head(top_n).sum() / total_gp
    bottom30_gp_share = sku_gp.tail(bottom_n).sum() / total_gp
    neg_skus = sku_gp[sku_gp < 0]

    quarantine = pd.read_csv(EDA_OUTPUT_DIR / "sku_quarantine_candidates.csv")

    returns_with_products = returns.merge(products[["product_id", "size"]], on="product_id", how="left")
    return_reason = (
        returns.groupby("return_reason")
        .agg(rows=("return_id", "count"), refund_amount=("refund_amount", "sum"))
        .reset_index()
        .sort_values("refund_amount", ascending=False)
    )
    return_reason["row_share"] = return_reason["rows"] / return_reason["rows"].sum()
    wrong_row = return_reason.loc[return_reason["return_reason"].eq("wrong_size")].iloc[0]

    fulfilled_qty_by_size = f.groupby("size")["quantity"].sum()
    return_qty_by_size = returns_with_products.groupby("size")["return_quantity"].sum()
    size_return_rate = (return_qty_by_size / fulfilled_qty_by_size).reindex(["S", "M", "L", "XL"])
    wrong_size_share_by_size = (
        returns_with_products.assign(is_wrong_size=returns_with_products["return_reason"].eq("wrong_size"))
        .groupby("size")
        .agg(return_rows=("return_id", "count"), wrong_size_rows=("is_wrong_size", "sum"))
    )
    size_wrong_size_share = (
        wrong_size_share_by_size["wrong_size_rows"] / wrong_size_share_by_size["return_rows"]
    ).reindex(["S", "M", "L", "XL"])

    committed_orders = data["orders"].loc[
        data["orders"]["order_status"].isin(COMMITTED_STATUSES), "order_id"
    ].nunique()
    return_rate_order = returns["order_id"].nunique() / committed_orders

    metrics = ChartMetrics(
        net_fulfilled_sales_b=net_fulfilled_sales / 1e9,
        net_fulfilled_margin=overall_margin,
        promo_margin=promo_margin,
        nonpromo_margin=nonpromo_margin,
        margin_gap_pp=(nonpromo_margin - promo_margin) * 100,
        promo_discount_burn_m=promo_discount / 1e6,
        offpeak_discount_burn_m=offpeak_discount / 1e6,
        offpeak_discount_share_fulfilled=offpeak_discount / promo_discount,
        promo_existing_share_recent=existing_share,
        promo_new_share_recent=new_share,
        corr_promo_net_revenue=daily["promo_rate"].corr(daily["net_rev"]),
        corr_promo_buyer_count=daily["promo_rate"].corr(daily["unique_buyers"]),
        revenue_day_ratio=revenue_day_ratio,
        sessions_day_ratio=sessions_day_ratio,
        peak_promo_rate=peak_promo_rate,
        offpeak_promo_rate=offpeak_promo_rate,
        streetwear_stockout_days=float(priority_stockouts.get("Streetwear", np.nan)),
        outdoor_stockout_days=float(priority_stockouts.get("Outdoor", np.nan)),
        top20_gp_share=top20_gp_share,
        bottom30_gp_share=bottom30_gp_share,
        negative_sku_count=int((sku_gp < 0).sum()),
        negative_sku_loss_m=-neg_skus.sum() / 1e6,
        quarantine_sku_count=len(quarantine),
        quarantine_loss_m=-quarantine["net_gp"].sum() / 1e6,
        wrong_size_returns=int(wrong_row["rows"]),
        wrong_size_reason_share=float(wrong_row["row_share"]),
        wrong_size_refund_m=float(wrong_row["refund_amount"]) / 1e6,
        return_rate_order=return_rate_order,
        size_return_rate_spread_pp=(size_return_rate.max() - size_return_rate.min()) * 100,
        wrong_size_share_spread_pp=(size_wrong_size_share.max() - size_wrong_size_share.min()) * 100,
    )
    return metrics, campaign_mix, return_reason, size_wrong_size_share


def plot_promo_capital_trap(metrics: ChartMetrics, campaign_mix: pd.DataFrame, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(9.85, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.28, 1.20], wspace=0.52)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel A: margin economics.
    ax = axes[0]
    labels = ["Có KM", "Không KM"]
    values = [
        metrics.promo_margin * 100,
        metrics.nonpromo_margin * 100,
    ]
    y = np.arange(len(labels))
    ax.barh(y, values, color=[COLORS["red"], COLORS["emerald"]], height=0.52)
    ax.axvline(0, color=CLR_LABEL, lw=0.9)
    ax.axvline(metrics.net_fulfilled_margin * 100, color=COLORS["blue"], linestyle="--", lw=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-18, 24)
    ax.set_xlabel("Biên lợi nhuận net fulfilled")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for yi, val in zip(y, values):
        if val < 0:
            ax.text(val / 2, yi, f"{val:+.1f}%", ha="center", va="center", color="white", fontsize=9.4, fontweight="bold")
        else:
            ax.text(val + 0.8, yi, f"{val:+.1f}%", ha="left", va="center", color=CLR_TITLE, fontsize=9.2)
    ax.text(
        metrics.net_fulfilled_margin * 100 + 0.5,
        1.26,
        f"toàn bộ {metrics.net_fulfilled_margin * 100:.1f}%",
        ha="left",
        va="center",
        fontsize=8.8,
        color=COLORS["blue"],
    )
    annotate_box(ax, f"chênh {metrics.margin_gap_pp:.1f}pp", 0.07, 0.50, color=COLORS["violet"], fontsize=9.2)
    add_panel_label(ax, "A. Rò rỉ biên lợi nhuận")
    style_report_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)

    # Panel B: recent-campaign customer mix.
    ax = axes[1]
    plot_mix = campaign_mix.sort_values("start_date", ascending=False).head(10).copy()
    y = np.arange(len(plot_mix))
    existing_pct = plot_mix["existing_share"].to_numpy() * 100
    new_pct = plot_mix["new_share"].to_numpy() * 100
    ax.barh(y, existing_pct, color=COLORS["red"], height=0.58)
    ax.barh(y, new_pct, left=existing_pct, color=COLORS["emerald"], height=0.58)
    ax.set_xlim(0, 100)
    ax.set_yticks(y, plot_mix["short_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Tỷ trọng net sales khuyến mãi")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.text(
        0.04,
        0.98,
        f"10 chiến dịch\n"
        f"{metrics.promo_existing_share_recent * 100:.1f}% khách cũ\n"
        f"{metrics.promo_new_share_recent * 100:.1f}% khách mới",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=CLR_TITLE,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=CLR_GRID, linewidth=0.9, alpha=0.96),
    )
    annotate_box(
        ax,
        f"r(khách,promo)={metrics.corr_promo_buyer_count:+.3f}\n"
        f"r(doanh thu,promo)={metrics.corr_promo_net_revenue:+.3f}",
        0.51,
        0.98,
        color=CLR_TITLE,
        fontsize=8.8,
    )
    add_panel_label(ax, "B. 10 chiến dịch gần nhất")
    style_report_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", labelsize=8.4)
    ax.xaxis.label.set_size(8.8)
    ax.spines["left"].set_visible(False)

    # Panel C: seasonal timing.
    ax = axes[2]
    x = np.array([0, 1])
    ax.bar(x, [metrics.revenue_day_ratio, 1.0], width=0.46, color=[COLORS["emerald"], "#CBD5E1"])
    ax.set_xticks(x, ["Cao điểm\nT4-T6", "Thấp điểm"])
    ax.set_ylim(0, 2.25)
    ax.set_ylabel("Chỉ số doanh thu/ngày")
    for xi, val in zip(x, [metrics.revenue_day_ratio, 1.0]):
        ax.text(xi, val + 0.06, f"{val:.2f}x", ha="center", va="bottom", fontsize=9.4, color=CLR_TITLE)
    ax2 = ax.twinx()
    promo_rates = [metrics.peak_promo_rate * 100, metrics.offpeak_promo_rate * 100]
    ax2.plot(x, promo_rates, marker="o", color=COLORS["violet"], linewidth=2.4)
    ax2.set_ylim(0, 55)
    ax2.set_ylabel("Tỷ lệ khuyến mãi", color=COLORS["violet"])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.tick_params(axis="y", colors=COLORS["violet"], labelsize=8.6)
    for xi, val in zip(x, promo_rates):
        ax2.text(xi, val + 2.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.4, color=COLORS["violet"])
    ax.text(
        0.67,
        0.08,
        f"{metrics.offpeak_discount_share_fulfilled * 100:.1f}% giảm giá\n"
        "ở thấp điểm\n"
        "Ngày-SKU hết hàng\n"
        f"Streetwear {metrics.streetwear_stockout_days / 1000:.1f}k\n"
        f"Outdoor {metrics.outdoor_stockout_days / 1000:.1f}k",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=CLR_TITLE,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor=CLR_GRID, linewidth=0.8, alpha=0.96),
    )
    add_panel_label(ax, "C. Cầu cao điểm, khuyến mãi thấp điểm")
    style_report_axes(ax, grid_axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["violet"])

    fig.suptitle(
        "Bẫy vốn khuyến mãi: biên âm, khách cũ, lệch mùa",
        x=0.02,
        y=0.96,
        ha="left",
        fontsize=13.8,
        fontweight="bold",
        color=CLR_TITLE,
    )
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.065, right=0.965, wspace=0.52)
    path = output_dir / "fig_report_insight_a_promo_capital_trap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_portfolio_drag(
    metrics: ChartMetrics,
    return_reason: pd.DataFrame,
    size_wrong_size_share: pd.Series,
    output_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(9.85, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.04, 1.12], wspace=0.55)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel A: SKU contribution concentration.
    ax = axes[0]
    labels = ["Top 20% SKU", "Bottom 30% SKU"]
    values = [metrics.top20_gp_share * 100, metrics.bottom30_gp_share * 100]
    y = np.arange(len(labels))
    ax.barh(y, values, color=[COLORS["emerald"], COLORS["red"]], height=0.50)
    ax.axvline(0, color=CLR_LABEL, lw=0.9)
    ax.axvline(100, color=CLR_MUTED, linestyle="--", lw=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-40, 135)
    ax.set_xlabel("Đóng góp net GP lũy kế")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for yi, val in zip(y, values):
        if val < 0:
            ax.text(2.5, yi, f"phá {abs(val):.1f}% GP", ha="left", va="center", color=COLORS["red"], fontsize=9.0, fontweight="bold")
        else:
            ax.text(val + 2.5, yi, f"+{val:.1f}%", ha="left", va="center", fontsize=9.2, color=CLR_TITLE)
    ax.text(
        0.03,
        0.92,
        f"{metrics.negative_sku_count} SKU GP âm\n"
        f"lỗ {metrics.negative_sku_loss_m:,.1f}M VND\n"
        "giai đoạn 2012-2022",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.9,
        color=CLR_TITLE,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor=CLR_GRID, linewidth=0.9),
    )
    add_panel_label(ax, "A. GP tập trung")
    style_report_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)

    # Panel B: return reason Pareto.
    ax = axes[1]
    reason_plot = return_reason.head(5).copy().sort_values("refund_amount")
    reason_labels = {
        "wrong_size": "sai size",
        "defective": "lỗi sản phẩm",
        "not_as_described": "khác mô tả",
        "changed_mind": "đổi ý",
        "late_delivery": "giao trễ",
    }
    colors = [COLORS["red"] if reason == "wrong_size" else COLORS["cyan"] for reason in reason_plot["return_reason"]]
    ax.barh(reason_plot["return_reason"].map(reason_labels), reason_plot["refund_amount"] / 1e6, color=colors, height=0.48)
    ax.set_xlim(0, reason_plot["refund_amount"].max() / 1e6 * 1.38)
    ax.set_xlabel("Hoàn tiền (triệu VND)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}M"))
    wrong = return_reason.loc[return_reason["return_reason"].eq("wrong_size")].iloc[0]
    ax.text(
        wrong["refund_amount"] / 1e6 - 7,
        reason_plot.index[reason_plot["return_reason"].eq("wrong_size")][0],
        f"{metrics.wrong_size_refund_m:.1f}M\n{metrics.wrong_size_reason_share * 100:.1f}% lý do",
        ha="right",
        va="center",
        fontsize=9.0,
        color="white",
        fontweight="semibold",
    )
    add_panel_label(ax, "B. Thất thoát do trả hàng")
    style_report_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)

    # Panel C: wrong-size flatness diagnostic.
    ax = axes[2]
    sizes = ["S", "M", "L", "XL"]
    rates = size_wrong_size_share.reindex(sizes) * 100
    x = np.arange(len(sizes))
    ax.axhspan(rates.min(), rates.max(), color=COLORS["red"], alpha=0.08, zorder=0)
    ax.plot(x, rates, color=COLORS["violet"], linewidth=1.8, marker="o", markersize=6)
    ax.scatter(x, rates, color=[COLORS["blue"], COLORS["flame"], COLORS["emerald"], COLORS["violet"]], s=52, zorder=3)
    ax.set_xticks(x, sizes)
    ax.set_ylim(0, 42.0)
    ax.set_ylabel("Tỷ trọng sai size")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    for xi, val in zip(x, rates):
        ax.text(xi, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.0, color=CLR_TITLE)
    annotate_box(
        ax,
        f"chênh chỉ {metrics.wrong_size_share_spread_pp:.2f}pp\nlỗi hướng dẫn size/PDP",
        0.04,
        0.27,
        color=COLORS["violet"],
        fontsize=9.2,
    )
    add_panel_label(ax, "C. Không do một size")
    style_report_axes(ax, grid_axis="y")

    fig.suptitle(
        "Danh mục kéo tụt GP: GP tập trung, sai size gây thất thoát",
        x=0.02,
        y=0.96,
        ha="left",
        fontsize=13.8,
        fontweight="bold",
        color=CLR_TITLE,
    )
    fig.subplots_adjust(top=0.78, bottom=0.19, left=0.07, right=0.965, wspace=0.55)
    path = output_dir / "fig_report_insight_b_portfolio_drag.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def write_metric_snapshot(metrics: ChartMetrics, output_dir: Path) -> Path:
    snapshot = pd.DataFrame([asdict(metrics)]).T.reset_index()
    snapshot.columns = ["metric", "value"]
    path = output_dir / "report_chart_metric_snapshot.csv"
    snapshot.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final-report EDA charts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--copy-to-report-revision",
        action="store_true",
        help="Also copy PNGs to The_4_Outliers_VinDatathon/outputs/report_revision.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_clean_data()
    metrics, campaign_mix, return_reason, size_wrong_size_share = compute_metrics(data)
    path_a = plot_promo_capital_trap(metrics, campaign_mix, args.output_dir)
    path_b = plot_portfolio_drag(metrics, return_reason, size_wrong_size_share, args.output_dir)
    snapshot = write_metric_snapshot(metrics, args.output_dir)

    if args.copy_to_report_revision:
        target_dir = EDA_OUTPUT_DIR / "report_revision"
        target_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(path_a, target_dir / "fig_insight_a_promo_capital_trap.png")
        shutil.copy2(path_b, target_dir / "fig_insight_b_portfolio_drag.png")
        shutil.copy2(path_a.with_suffix(".svg"), target_dir / "fig_insight_a_promo_capital_trap.svg")
        shutil.copy2(path_b.with_suffix(".svg"), target_dir / "fig_insight_b_portfolio_drag.svg")

    print(f"Saved {path_a}")
    print(f"Saved {path_b}")
    print(f"Saved {snapshot}")


if __name__ == "__main__":
    main()
