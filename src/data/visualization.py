from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px


def gene_expression_plot(
    expr_df: pd.DataFrame,
    save_path: Path,
    title: str = "",
    color_discrete_sequence: Iterable[str] = None,
    gene_expr_col: str = "gene_expr",
    gene_expr_level: str = "gene_expr_level",
):
    """
        Plots the expression values of a certain gene amongst many samples.
        The first and last deciles are colored
            differently.

    Args:
        expr_df: Contains the log2 counts of gene expression, with as many
        rows as samples involved.
        save_path: File to save the generated plot to.
        title: Plot title
        color_discrete_sequence: List of three elements, each representing a
            color for each category in the graph.
        gene_expr_col: Column containing expression values.
        gene_expr_level: Column name of the new column containing the
            expression levels.
    """
    if not color_discrete_sequence:
        color_discrete_sequence = ["red", "blue", "green"]
    expr_df = deepcopy(expr_df)
    expr_df.sort_values(gene_expr_col, inplace=True)

    # 2. Plot and save
    fig = px.bar(
        expr_df,
        x=expr_df.index,
        y=gene_expr_col,
        color=gene_expr_level,
        title=title,
        text=gene_expr_col,
        color_discrete_sequence=color_discrete_sequence,
    )
    fig = fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig = fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")

    # 2.1. Image format depends on file name extension. Only pdf and HTML
    # accepted for now.
    if save_path.suffix == ".pdf":
        fig.write_image(str(save_path))
    elif save_path.suffix == ".html":
        fig.write_html(str(save_path))
    else:
        raise ValueError(
            f"Save file had suffix {save_path.suffix},"
            "but only .pdf and .html are possible."
        )

    return fig
