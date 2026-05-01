"""
DATATHON 2026 — Vivid Slate Design System
==========================================
Cách dùng:
    from theme import apply_theme, style_ax, COLORS, C, get_palette, colors_for

    apply_theme()          # gọi 1 lần ở đầu notebook là xong

    fig, ax = plt.subplots()
    ax.plot(x, y)
    style_ax(ax, title="Tiêu đề chart", subtitle="Mô tả ngắn")
    plt.savefig("chart.png")
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────

COLORS = {
    "blue":    "#2563EB",   # primary — revenue, main series
    "flame":   "#EA580C",   # secondary — COGS, cost
    "emerald": "#16A34A",   # positive — growth, profit
    "red":     "#DC2626",   # negative — return, loss, alert
    "violet":  "#7C3AED",   # accent — promo, annotation
    "cyan":    "#0891B2",   # accent — traffic, secondary signal
}

# Semantic cycle is reserved for charts where the color itself has meaning.
SEMANTIC_C = list(COLORS.values())

# Categorical palette for segment/category/channel charts.
# Keep this separate from COLORS so arbitrary categories do not inherit
# business meaning like "red = alert" or "emerald = positive".
CATEGORICAL = [
    "#2563EB",  # blue
    "#EA580C",  # flame
    "#16A34A",  # emerald
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#DB2777",  # rose
    "#CA8A04",  # gold
    "#4F46E5",  # indigo
    "#0D9488",  # teal
    "#9333EA",  # purple
    "#65A30D",  # lime
    "#475569",  # slate
]

# Shorthand list for default color cycles and category loops.
C = CATEGORICAL

# Stable domain palettes. Use these when the same label appears in multiple
# charts and must keep the same color.
SEGMENT_COLORS = {
    "Trendy": "#2563EB",
    "Activewear": "#EA580C",
    "Standard": "#16A34A",
    "Everyday": "#7C3AED",
    "Balanced": "#0891B2",
    "Performance": "#DB2777",
    "Premium": "#CA8A04",
    "All-weather": "#4F46E5",
}

CATEGORY_COLORS = {
    "Streetwear": "#2563EB",
    "Casual": "#EA580C",
    "GenZ": "#16A34A",
    "Outdoor": "#7C3AED",
}

TRAFFIC_SOURCE_COLORS = {
    "organic_search": "#2563EB",
    "paid_search": "#EA580C",
    "social_media": "#16A34A",
    "email_campaign": "#7C3AED",
    "referral": "#0891B2",
    "direct": "#DB2777",
}

ORDER_STATUS_COLORS = {
    "delivered": "#16A34A",
    "shipped": "#0891B2",
    "paid": "#2563EB",
    "created": "#475569",
    "returned": "#DC2626",
    "cancelled": "#EA580C",
}

DOMAIN_COLOR_MAPS = {
    "segment": SEGMENT_COLORS,
    "category": CATEGORY_COLORS,
    "traffic_source": TRAFFIC_SOURCE_COLORS,
    "order_status": ORDER_STATUS_COLORS,
}

# Sequential palettes (dùng cho heatmap, choropleth)
SEQ_BLUE  = ["#DBEAFE", "#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A8A"]
SEQ_AMBER = ["#FEF3C7", "#FCD34D", "#F59E0B", "#D97706", "#92400E"]

# Diverging (dùng cho growth rate, so sánh YoY)
DIV = ["#DC2626", "#F87171", "#F9FAFB", "#60A5FA", "#2563EB"]

# Neutrals
BG_FIGURE  = "#FFFFFF"
BG_AXES    = "#F9FAFB"
CLR_GRID   = "#E5E7EB"
CLR_SPINE  = "#D1D5DB"
CLR_LABEL  = "#374151"
CLR_TITLE  = "#111827"
CLR_MUTED  = "#9CA3AF"


def get_palette(n, palette=None):
    """
    Return n categorical colors without repeating the short semantic palette.

    Use this for arbitrary groups such as segment, category, city, color,
    acquisition channel, or top-N bars. For Revenue/COGS/growth/alert charts,
    keep using explicit COLORS keys.
    """
    if n <= 0:
        return []

    base = list(CATEGORICAL if palette is None else palette)
    if n <= len(base):
        return base[:n]

    colors = base[:]
    used = {c.lower() for c in colors}
    # Generate evenly spaced fallback hues only when the curated palette is
    # exhausted. This avoids silent repeats in high-cardinality charts.
    hues = np.linspace(0, 1, n * 2, endpoint=False)
    hsv = np.column_stack([hues, np.full_like(hues, 0.68), np.full_like(hues, 0.82)])
    generated = [mpl.colors.to_hex(rgb) for rgb in mpl.colors.hsv_to_rgb(hsv)]

    for color in generated:
        if color.lower() not in used:
            colors.append(color)
            used.add(color.lower())
        if len(colors) >= n:
            break

    return colors[:n]


def get_color_map(labels, palette=None, known_map=None, sort=False):
    """
    Build a stable {label: color} mapping.

    known_map can be a dict or one of: "segment", "category",
    "traffic_source", "order_status".
    """
    unique_labels = list(dict.fromkeys(labels))
    if sort:
        unique_labels = sorted(unique_labels)

    if isinstance(known_map, str):
        known = DOMAIN_COLOR_MAPS.get(known_map, {})
    else:
        known = known_map or {}

    mapping = {}
    used = set()
    for label in unique_labels:
        if label in known:
            color = known[label]
        else:
            color = known.get(str(label))
        if color:
            mapping[label] = color
            used.add(color.lower())

    fallback = [c for c in get_palette(len(unique_labels) + len(used), palette)
                if c.lower() not in used]
    fallback_iter = iter(fallback)
    for label in unique_labels:
        if label not in mapping:
            mapping[label] = next(fallback_iter)

    return mapping


def colors_for(labels, palette=None, known_map=None, sort=False):
    """Return colors aligned to labels."""
    color_map = get_color_map(labels, palette=palette, known_map=known_map, sort=sort)
    return [color_map[label] for label in labels]


def theme_palette(n):
    """Backward-compatible helper for notebooks that call theme_palette(n)."""
    return get_palette(n)


# ── Theme ────────────────────────────────────────────────────────────────────

def apply_theme():
    """Áp dụng Vivid Slate theme. Gọi 1 lần ở đầu notebook."""
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":      BG_FIGURE,
        "figure.figsize":        (10, 5),
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     BG_FIGURE,

        # Axes
        "axes.facecolor":        BG_AXES,
        "axes.edgecolor":        CLR_SPINE,
        "axes.linewidth":        0.8,
        "axes.labelcolor":       CLR_LABEL,
        "axes.labelsize":        11,
        "axes.labelpad":         8,
        "axes.titlesize":        13,
        "axes.titleweight":      "medium",
        "axes.titlecolor":       CLR_TITLE,
        "axes.titlelocation":    "left",
        "axes.titlepad":         10,
        "axes.spines.right":     False,
        "axes.spines.top":       False,
        "axes.prop_cycle":       plt.cycler(color=C),

        # Grid
        "axes.grid":             True,
        "grid.color":            CLR_GRID,
        "grid.linewidth":        0.5,
        "grid.alpha":            1.0,
        "axes.grid.axis":        "y",           # chỉ grid ngang

        # Ticks
        "xtick.color":           CLR_LABEL,
        "ytick.color":           CLR_LABEL,
        "xtick.labelsize":       10,
        "ytick.labelsize":       10,
        "xtick.major.size":      4,
        "ytick.major.size":      0,            # ẩn y-tick marks
        "xtick.direction":       "out",
        "ytick.direction":       "out",

        # Text
        "text.color":            CLR_TITLE,
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Helvetica Neue", "Arial", "DejaVu Sans"],

        # Legend
        "legend.frameon":        False,
        "legend.fontsize":       10,
        "legend.labelcolor":     CLR_LABEL,
        "legend.handlelength":   1.2,
        "legend.handleheight":   0.8,

        # Lines
        "lines.linewidth":       2.0,
        "lines.markersize":      5,

        # Patches (bar, area)
        "patch.linewidth":       0,
    })
    print("Vivid Slate theme applied.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def style_ax(ax, title="", subtitle="", xlabel="", ylabel="", yformat=None):
    """
    Chuẩn hóa một axes sau khi plot xong.

    Params:
        ax       — matplotlib Axes
        title    — tiêu đề chính (bold, trái)
        subtitle — dòng nhỏ bên dưới title (muted, trái)
        xlabel   — nhãn trục X
        ylabel   — nhãn trục Y
        yformat  — format string cho y-tick, vd: '{x:,.0f}' hoặc '{x:.1%}'
    """
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="medium",
                     color=CLR_TITLE, pad=10)
    if subtitle:
        # Đặt subtitle ngay dưới title
        ax.text(0, 1.035, subtitle, transform=ax.transAxes,
                fontsize=9.5, color=CLR_MUTED, va="bottom", ha="left")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if yformat:
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: yformat.format(x=x))
        )

    # Xóa bottom spine nếu không cần thiết
    ax.spines["bottom"].set_color(CLR_SPINE)
    ax.spines["left"].set_visible(False)


def annotate_bar(ax, rects, fmt="{:.0f}", offset=3, color=None):
    """
    Thêm label giá trị lên đầu mỗi cột bar chart.

    Dùng: annotate_bar(ax, ax.patches)
    """
    _color = color or CLR_LABEL
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, color=_color,
        )


def add_insight(ax, x, y, text, color=None, arrow=True):
    """
    Annotate một điểm nổi bật trên chart với mũi tên.

    Dùng cho Prescriptive/Diagnostic storytelling.
    Vd: add_insight(ax, '2020-04', 15000, 'COVID spike\\n+42% YoY')
    """
    _color = color or COLORS["violet"]
    props = dict(arrowstyle="->", color=_color, lw=1.2) if arrow else None
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=9,
        color=_color,
        arrowprops=props,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=_color, linewidth=0.8, alpha=0.9),
    )


def make_legend(ax, labels, colors=None, loc="upper left", ncol=1):
    """
    Custom legend với ô vuông màu (thay vì đường mặc định).

    Dùng sau khi plot xong, trước savefig.
    """
    _colors = colors or get_palette(len(labels))
    patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(_colors, labels)
    ]
    ax.legend(handles=patches, loc=loc, ncol=ncol,
              frameon=False, fontsize=10)


def heatmap(data, row_labels, col_labels, ax=None, cmap=None,
            fmt=".1f", title="", subtitle=""):
    """
    Vẽ heatmap chuẩn với annotation giá trị bên trong ô.

    Params:
        data        — 2D numpy array
        row_labels  — list nhãn hàng
        col_labels  — list nhãn cột
        cmap        — colormap (default: Blues)
    """
    if ax is None:
        _, ax = plt.subplots()

    _cmap = cmap or mpl.colors.LinearSegmentedColormap.from_list(
        "vs_blue", ["#DBEAFE", "#2563EB"]
    )

    im = ax.imshow(data, cmap=_cmap, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.grid(False)

    # Annotate từng ô
    thresh = data.max() / 2
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center", fontsize=9,
                    color="white" if val > thresh else CLR_TITLE)

    style_ax(ax, title=title, subtitle=subtitle)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(bottom=False, left=False)
    return ax, im


# ── Plotly template (optional) ───────────────────────────────────────────────

def get_plotly_template():
    """
    Trả về Plotly template tương ứng với Vivid Slate.
    Dùng: fig = px.line(..., template=get_plotly_template())
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        template = go.layout.Template(
            layout=dict(
                colorway=C,
                paper_bgcolor=BG_FIGURE,
                plot_bgcolor=BG_AXES,
                font=dict(family="Helvetica Neue, Arial", color=CLR_LABEL, size=11),
                title=dict(font=dict(size=14, color=CLR_TITLE), x=0, xanchor="left"),
                xaxis=dict(showgrid=False, linecolor=CLR_SPINE,
                           tickfont=dict(size=10), zeroline=False),
                yaxis=dict(gridcolor=CLR_GRID, gridwidth=0.5, linewidth=0,
                           tickfont=dict(size=10), zeroline=False),
                legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                            font=dict(size=10)),
                margin=dict(l=60, r=20, t=55, b=50),
                hoverlabel=dict(bgcolor="white", bordercolor=CLR_SPINE,
                                font=dict(size=11)),
            )
        )
        pio.templates["vivid_slate"] = template
        return "vivid_slate"
    except ImportError:
        print("Plotly chưa được cài. Chạy: pip install plotly")
        return None
