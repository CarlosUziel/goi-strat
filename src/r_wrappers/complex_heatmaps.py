"""
Wrappers for R package ComplexHeatmap

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_complex_heatmaps = importr("ComplexHeatmap")
pdf = ro.r("pdf")
dev_off = ro.r("dev.off")


def complex_heatmap(
    counts_matrix: pd.DataFrame,
    save_path: Path,
    width: int = 10,
    height: int = 20,
    heatmap_legend_side: str = "right",
    annotation_legend_side: str = "right",
    **kwargs: Any,
) -> None:
    """Create a complex heatmap visualization and save it to a file.

    This function creates a heatmap visualization from a dataframe of numeric values,
    with many customization options for clustering, color scales, annotations, etc.

    Args:
        counts_matrix: A pandas dataframe containing the numeric values for
            the heatmap. Rows typically represent features (genes, proteins, etc.)
            and columns represent samples or conditions.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        heatmap_legend_side: Position of the heatmap legend ("right", "left",
            "bottom", or "top").
        annotation_legend_side: Position of the annotation legend ("right", "left",
            "bottom", or "top").
        **kwargs: Additional arguments to pass to the Heatmap function.
            Common parameters include:
            - name: Name of the heatmap, used as the title for the color legend.
            - cluster_rows: Whether to cluster rows (default: TRUE).
            - cluster_columns: Whether to cluster columns (default: TRUE).
            - show_row_names: Whether to show row names (default: TRUE).
            - show_column_names: Whether to show column names (default: TRUE).
            - row_names_side: Side to place the row names ("left" or "right").
            - column_names_side: Side to place the column names ("top" or "bottom").
            - top_annotation: Annotation to add to the top of the heatmap.
            - right_annotation: Annotation to add to the right of the heatmap.
            - col: Color mapping function for the heatmap.
            - row_split: Splits rows into different groups.
            - column_split: Splits columns into different groups.

    References:
        https://rdrr.io/bioc/ComplexHeatmap/man/Heatmap.html
    """
    # 0. Compute heatmap
    mat = ro.r("as.matrix")(pd_df_to_rpy2_df(counts_matrix))
    mat.rownames = ro.StrVector(counts_matrix.index.tolist())
    ht = r_complex_heatmaps.Heatmap(
        mat,
        **kwargs,
        row_names_max_width=ro.r("unit")(25, "cm"),
        column_names_max_height=ro.r("unit")(25, "cm"),
    )

    # 1. Save heatmap
    pdf(str(save_path), width=width, height=height)
    r_complex_heatmaps.draw(
        ht,
        heatmap_legend_side=heatmap_legend_side,
        annotation_legend_side=annotation_legend_side,
        merge_legend=True,
    )
    dev_off()


def heatmap_annotation(
    df: pd.DataFrame, col: Optional[Dict[str, Dict[str, str]]] = ro.NULL, **kwargs: Any
) -> Any:
    """Create a heatmap annotation object for annotating heatmap columns.

    This function creates a HeatmapAnnotation object that can be used to add
    annotations to the columns (or rows) of a heatmap. These annotations can
    represent categorical variables, continuous variables, or other data types.

    Args:
        df: A DataFrame where each column will be treated as a simple annotation.
            The DataFrame must have column names. Each column creates one row of
            annotation in the heatmap.
        col: A dictionary of dictionaries, where each element is a column name
            of df containing a mapping of column values to colors. For example:
            {"condition": {"treatment": "red", "control": "blue"}}
        **kwargs: Additional arguments to pass to the HeatmapAnnotation function.
            Common parameters include:
            - name: Name of the annotation, used as the title for legends.
            - show_legend: Whether to show the annotation legend.
            - show_annotation_name: Whether to show the annotation name.
            - annotation_name_side: Side to place annotation names.
            - annotation_legend_param: List of parameters for annotation legends.

    Returns:
        Any: A HeatmapAnnotation object that can be passed to the complex_heatmap
        function as top_annotation, bottom_annotation, etc.

    Raises:
        ValueError: If any keys in col are not column names in df.

    References:
        https://rdrr.io/bioc/ComplexHeatmap/man/HeatmapAnnotation.html
    """
    if col is not ro.NULL:
        # 0. Process colors
        if [x for x in col.keys() if x not in df.columns]:
            raise ValueError("Some keys in col are not part of df")

        # todo: easier way to convert to StrVector with names? (ListVector in R)
        col = ro.ListVector(
            {
                k: ro.r(
                    "c(" + ", ".join([f'"{k}"="{v}"' for k, v in col[k].items()]) + ")"
                )
                for k in col.keys()
            }
        )

    return r_complex_heatmaps.HeatmapAnnotation(
        df=pd_df_to_rpy2_df(df), col=col, **kwargs
    )


def anno_barplot(values: Union[Iterable[float], pd.DataFrame], **kwargs: Any) -> Any:
    """Create a barplot annotation for a heatmap.

    This function creates a barplot annotation that can be added to a heatmap
    to show the distribution of a continuous variable across samples or features.

    Args:
        values: A vector of numeric values or a DataFrame. If the value is a DataFrame,
            columns of the DataFrame will be represented as stacked barplots. For stacked
            barplots, each row in the DataFrame should only contain values with the same
            sign (either all positive or all negative).
        **kwargs: Additional arguments to pass to the anno_barplot function.
            Common parameters include:
            - gp: Graphical parameters for the bars.
            - border: Whether to draw borders around bars.
            - bar_width: Width of bars.
            - height: Height of the annotation.
            - axis: Whether to add axis to the barplot.
            - axis_param: Parameters for the axis.
            - ylim: Y-limits for the barplot.

    Returns:
        Any: An annotation function that can be used in HeatmapAnnotation.

    References:
        https://rdrr.io/bioc/ComplexHeatmap/man/anno_barplot.html
    """
    values = (
        ro.r("as.matrix")(pd_df_to_rpy2_df(values))
        if isinstance(values, pd.DataFrame)
        else ro.FloatVector(values)
    )

    return r_complex_heatmaps.anno_barplot(values, **kwargs)
