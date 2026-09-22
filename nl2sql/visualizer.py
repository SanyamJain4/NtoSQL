"""
Auto-visualization: inspects the shape of a query result (row/column count,
dtypes) and picks a sensible chart type, mirroring what a human analyst
would do without being told explicitly. Falls back to a plain table when
no chart type clearly fits.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(5)
    return sample.str.match(r"^\d{4}-\d{2}(-\d{2})?$").all() if len(sample) else False


def choose_chart_type(df: pd.DataFrame) -> str:
    if df.empty:
        return "table"

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # Single numeric value overall -> just show the number as a table/KPI.
    # Check this before the "need >=2 columns" gate below, since a KPI result
    # is legitimately a single column, single row.
    if len(df) == 1 and len(numeric_cols) == 1 and len(df.columns) == 1:
        return "kpi"

    if len(df.columns) < 2 or not numeric_cols:
        return "table"

    # One non-numeric column that looks like a date + one numeric -> line chart (trend).
    if len(non_numeric_cols) == 1 and _is_datetime_like(df[non_numeric_cols[0]]) and len(numeric_cols) >= 1:
        return "line"

    # One categorical column + one numeric, few rows -> bar chart.
    if len(non_numeric_cols) == 1 and len(numeric_cols) == 1 and len(df) <= 30:
        return "bar"

    # Two categorical dimensions + one numeric -> could be a grouped bar or heatmap; default to bar of top rows.
    if len(non_numeric_cols) >= 2 and len(numeric_cols) == 1:
        return "grouped_bar"

    # Many rows, numeric -> scatter/histogram territory.
    if len(numeric_cols) >= 2 and len(df) > 10:
        return "scatter"

    return "table"


def render(df: pd.DataFrame, title: str, out_path: str) -> str:
    """Renders the chosen chart to out_path (PNG) and returns the chart type used."""
    chart_type = choose_chart_type(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if chart_type == "bar":
        cat_col, val_col = non_numeric_cols[0], numeric_cols[0]
        data = df.sort_values(val_col, ascending=False)
        ax.bar(data[cat_col].astype(str), data[val_col], color="#1D9E75")
        ax.set_xlabel(cat_col)
        ax.set_ylabel(val_col)
        plt.xticks(rotation=40, ha="right")

    elif chart_type == "line":
        x_col, val_col = non_numeric_cols[0], numeric_cols[0]
        data = df.sort_values(x_col)
        ax.plot(data[x_col].astype(str), data[val_col], marker="o", color="#378ADD")
        ax.set_xlabel(x_col)
        ax.set_ylabel(val_col)
        plt.xticks(rotation=40, ha="right")

    elif chart_type == "grouped_bar":
        cat_col, group_col, val_col = non_numeric_cols[0], non_numeric_cols[1], numeric_cols[0]
        pivot = df.pivot_table(index=cat_col, columns=group_col, values=val_col, aggfunc="sum").fillna(0)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(val_col)
        plt.xticks(rotation=40, ha="right")

    elif chart_type == "scatter":
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        ax.scatter(df[x_col], df[y_col], alpha=0.6, color="#D85A30")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    elif chart_type == "kpi":
        val = df.iloc[0, numeric_cols[0] == df.columns].values if False else df[numeric_cols[0]].iloc[0]
        ax.text(0.5, 0.5, f"{val:,.2f}" if isinstance(val, float) else str(val),
                fontsize=40, ha="center", va="center")
        ax.set_title(df.columns[df.columns.get_loc(numeric_cols[0])])
        ax.axis("off")

    else:  # table -- render as a matplotlib table for a consistent artifact
        ax.axis("off")
        tbl = ax.table(cellText=df.head(20).values, colLabels=df.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return chart_type
