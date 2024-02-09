"""
    Wrappers for R package ComplexHeatmap

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

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
    **kwargs,
):
    """
    Plots a heatmap of a dataframe.

    *ref docs: https://rdrr.io/bioc/ComplexHeatmap/man/Heatmap.html

    Args:
        counts_matrix: A pandas dataframe containing the numeric values for
            the heatmap.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure

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
    df: pd.DataFrame, col: Optional[Dict[str, Dict[str, str]]] = ro.NULL, **kwargs
):
    """
    Creates a heatmap annotation object used to annotate heatmap columns.

    *ref docs in https://rdrr.io/bioc/ComplexHeatmap/man/HeatmapAnnotation.html

    Args:
        df: A ann_df frame where each column will be treated as a simple
            annotation. The ann_df frame must have column names.
        col: A dictionary of dictionaries, where each element is a column
            name of df containing a mapping of column values and colors.
            See SingleAnnotation for how to set colors.
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


def anno_barplot(values: Union[Iterable[float], pd.DataFrame], **kwargs):
    """
    Using barplot as annotation.

    *ref docs in https://rdrr.io/github/eilslabs/ComplexHeatmap/man/anno_barplot.html

    Args:
        values: A vector of numeric values. If the value is a matrix, columns of the
            matrix will be represented as stacked barplots. Note for stacked barplots,
            each row in the matrix should only contain values with same sign (either
            all positive or all negative).
    """
    values = (
        ro.r("as.matrix")(pd_df_to_rpy2_df(values))
        if isinstance(values, pd.DataFrame)
        else ro.FloatVector(values)
    )

    return r_complex_heatmaps.anno_barplot(values, **kwargs)
