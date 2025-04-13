"""Visualization utilities for gene expression and genomic data analysis.

This module provides functions to create and save visualizations of gene expression
data and other genomic analysis results using Plotly.
"""

from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px


def gene_expression_plot(
    expr_df: pd.DataFrame,
    save_path: Path,
    title: str = "",
    color_discrete_sequence: Optional[List[str]] = None,
    gene_expr_col: str = "gene_expr",
    gene_expr_level: str = "gene_expr_level",
) -> px.bar:
    """Create and save a bar plot of gene expression values across samples.

    Generates a bar plot showing gene expression values for each sample, with
    bars colored according to expression level categories (low, mid, high).

    Args:
        expr_df: DataFrame containing gene expression values with samples as rows
        save_path: Path where the plot will be saved
        title: Title for the plot
        color_discrete_sequence: List of three colors for low, mid, and high expression
            levels, defaults to ["red", "blue", "green"]
        gene_expr_col: Column name containing expression values
        gene_expr_level: Column name containing expression level categories

    Returns:
        px.bar: Plotly bar chart figure object

    Raises:
        ValueError: If save_path has an extension other than .pdf or .html

    Note:
        The function assumes expr_df has already been processed to contain
        expression level categories. If not, use gene_expression_levels()
        from data.utils first.
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
