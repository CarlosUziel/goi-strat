#!/bin/bash

mamba create -n bioinfo -c nvidia -c rapidsai -c conda-forge -c bioconda -y \
rpy2 typer pydantic pandas umap-learn plotly python-kaleido matplotlib-base \
jupyterlab ipywidgets tqdm autoflake black black-jupyter pylint flake8 \
bioconductor-tcgabiolinks bioconductor-deseq2 bioconductor-complexheatmap \
bioconductor-sesame bioconductor-sesamedata \
bioconductor-apeglm bioconductor-enhancedvolcano r-pheatmap r-venndiagram \
r-rcolorbrewer bioconductor-vsn bioconductor-annotationdbi r-devtools \
r-ashr r-gprofiler2 r-irkernel bioconductor-pathview bioconductor-methylkit \
bioconductor-minfi bioconductor-limma r-readxl r-msigdbr r-ggridges \
bioconductor-dmrcate bioconductor-dmrcatedata r-ggnewscale trim-galore \
bioconductor-annotatr bioconductor-minfidata r-scatterplot3d r-psych \
bioconductor-illuminahumanmethylation450kanno.ilmn12.hg19  r-earth \
bioconductor-txdb.hsapiens.ucsc.hg38.knowngene bioconductor-do.db \
bioconductor-clusterprofiler bioconductor-enrichplot r-ggupset \
bioconductor-dose bioconductor-fgsea r-europepmc bioconductor-reactomepa \
fastqc cutadapt star htseq multiqc samtools bamtools r-biocmanager \
seaborn=0.12.2 fastcluster r-paralleldist bioconductor-biocparallel \
r-languageserver radian r-httpgd bioconductor-ggtree kneed openpyxl \
shap  xgboost tblib r-jsonlite fish rich rich-cli natsort bioconductor-sva \
r-tidyverse r-purrr matplotlib-venn wordcloud biopython fonttools \
lifelines tabulate bioconductor-rnaseqpower upsetplot bedtools \
bioconductor-gsva lightgbm r-polychrome squarify colour picard \
bioconductor-cola bioconductor-simplifyenrichment sra-tools \
scikit-learn networkx nltk r-magick pygraphviz bismark pigz \
bioconductor-bsseq bioconductor-dss bioconductor-biseq deeptools \
bioconductor-txdb.mmusculus.ucsc.mm10.knowngene bioconductor-chipseeker \
bioconductor-rgreat bioconductor-org.mm.eg.db r-r.filesets python-docx \
pyarrow

conda activate bioinfo

pip install fasttreeshap
pip install node2vec

python -m nltk.downloader all

# install this in R:
Rscript -e 'library(devtools); install_github("alexeckert/parallelDist")'
Rscript -e 'library(devtools); install_github("markgene/maxprobes")'
Rscript -e 'library(devtools); install_github("Lothelab/CMScaller")'
Rscript -e 'library(devtools); install_github("moosa-r/rbioapi")'
Rscript -e 'library(devtools); install_github("al2na/methylKit", build_vignettes=FALSE, repos=BiocManager::repositories(), dependencies=TRUE)'
Rscript -e 'old.packages(repos="https://cloud.r-project.org/"); update.packages(ask=FALSE, repos="https://cloud.r-project.org/")'
Rscript -e 'remove.packages("DO.db"); BiocManager::install("DO.db"); BiocManager::install(ask=FALSE)'
Rscript -e 'BiocManager::install("BioinformaticsFMRP/TCGAbiolinksGUI.data"); BiocManager::install("BioinformaticsFMRP/TCGAbiolinks")'
Rscript -e 'BiocManager::install()' # to update all bioconductor packages
Rscript -e 'IRkernel::installspec()' # to install R kernel for Jupyter

# package fixes as of 04/08/2024
Rscript -e 'devtools::install_github('https://github.com/Bioconductor/BiocFileCache.git')'
Rscript -e 'remotes::install_version("matrixStats", version="1.1.0")'