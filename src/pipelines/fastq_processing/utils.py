"""
Utilities for processing FASTQ files in high-performance computing environments.

This module provides a collection of functions for processing next-generation sequencing
data files (primarily FASTQ) using bioinformatics tools in a high-performance computing
environment. The primary design goal is parallel processing of samples using SLURM job
scheduling.

The module includes functions for:
    - Quality control of sequencing data (FastQC)
    - Adapter trimming (Cutadapt, Trim Galore)
    - Read alignment (STAR, Bowtie2)
    - File format conversion (BAM to FASTQ, BAM to BED)
    - Methylation data processing (Bismark)
    - Post-alignment processing (HTSeq-count, Samtools)
    - Report generation (MultiQC)

Each function is designed to:
    1. Process a batch of input files (typically one per sample)
    2. Run a specific bioinformatics tool with configurable parameters
    3. Optionally submit the job to a SLURM cluster for parallel processing
    4. Organize outputs in a consistent directory structure

The module assumes a consistent file naming convention where sample IDs can be
extracted from file names, typically in the format: {sample_id}.{pair_n}.fastq.gz
"""

import logging
from collections import defaultdict
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from tqdm.rich import tqdm

from data.utils import run_cmd
from slurm.slurm_job_submitter import SlurmJobSubmitter


def run_fastqc(
    fastq_path: Path,
    fastqc_path: Path,
    fastqc_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run FastQC quality control on FASTQ files.

    FastQC is a popular tool to assess the quality of raw NGS data. This function
    processes multiple FASTQ files in parallel (one job per sample) using either
    direct execution or SLURM job submission.

    Expected naming scheme of fastq files is: {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path (Path): Path to directory containing FASTQ files.
        fastqc_path (Path): Path to directory where FastQC results will be stored.
        fastqc_kwargs (Dict[str, Any]): A dictionary of FastQC command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--threads": 8, "--quiet": ""}
        pattern (str): File name pattern that the FASTQ files to be processed should follow.
            Supports glob syntax relative to fastq_path.
        slurm_kwargs (Optional[Dict[str, Any]]): A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Results are written to the specified output directories.

    Examples:
        >>> run_fastqc(
        ...     fastq_path=Path("/data/raw_fastq"),
        ...     fastqc_path=Path("/data/fastqc_results"),
        ...     fastqc_kwargs={"--threads": "4"},
        ...     slurm_kwargs={"--mem": "16G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run fastqc for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = fastqc_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "fastqc",
                *sample_reads[:2],
                "-o",
                sample_out_path,
                *list(chain(*fastqc_kwargs.items())),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.fastqc.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.fastqc.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_cutadapt(
    fastq_path: Path,
    cutadapt_path: Path,
    fwd_adapter_file: Path,
    rv_adapter_file: Path,
    cutadapt_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run Cutadapt on FASTQ files to trim adapter sequences.

    Cutadapt finds and removes adapter sequences, primers, poly-A tails, and other
    unwanted sequences from high-throughput sequencing reads. This function processes
    multiple FASTQ files in parallel (one job per sample) using either direct execution
    or SLURM job submission.

    Expected naming scheme of FASTQ files is: {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path (Path): Path to directory containing FASTQ files.
        cutadapt_path (Path): Path to directory where trimmed reads will be stored.
        fwd_adapter_file (Path): Path to file containing forward adapter sequences. Will be
            passed to Cutadapt using the "file:" prefix.
        rv_adapter_file (Path): Path to file containing reverse adapter sequences. Will be
            used only for paired-end reads.
        cutadapt_kwargs (Dict[str, Any]): A dictionary of Cutadapt command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--quality-cutoff": "20", "--minimum-length": "50"}
        pattern (str): File name pattern that the FASTQ files to be processed should follow.
            Supports glob syntax relative to fastq_path.
        slurm_kwargs (Optional[Dict[str, Any]]): A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Trimmed FASTQ files are written to the specified output directory with
        naming format {sample_id}.{pair_n}.trimmed.fastq.gz.

    Examples:
        >>> run_cutadapt(
        ...     fastq_path=Path("/data/raw_fastq"),
        ...     cutadapt_path=Path("/data/trimmed_fastq"),
        ...     fwd_adapter_file=Path("/data/adapters/TruSeq_fwd.fa"),
        ...     rv_adapter_file=Path("/data/adapters/TruSeq_rev.fa"),
        ...     cutadapt_kwargs={"--quality-cutoff": "20", "--minimum-length": "50"},
        ...     slurm_kwargs={"--mem": "16G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run Cutadapt for each sample
    logging.info("Submitting SLURM jobs: ")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = cutadapt_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        read_1_trim = sample_out_path.joinpath(f"{sample_id}.1.trimmed.fastq.gz")
        read_2_trim = sample_out_path.joinpath(f"{sample_id}.2.trimmed.fastq.gz")

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "cutadapt",
                *sample_reads[:2],
                "-b",
                f"file:{fwd_adapter_file}",
                "-o",
                read_1_trim,
                *(
                    ["-B", f"file:{rv_adapter_file}", "-p", read_2_trim]
                    if len(sample_reads) > 1
                    else []
                ),
                *list(chain(*cutadapt_kwargs.items())),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.cutadapt.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.cutadapt.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_trim_galore(
    fastq_path: Path,
    trim_galore_path: Path,
    trim_galore_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run Trim Galore on FASTQ files to trim adapters and low-quality bases.

    Trim Galore is a wrapper around Cutadapt and FastQC to consistently apply adapter and
    quality trimming to FASTQ files, with extra functionality for bisulfite sequencing.
    This function processes multiple FASTQ files in parallel using direct execution
    or SLURM job submission.

    Expected naming scheme of FASTQ files is: {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path (Path): Path to directory containing FASTQ files.
        trim_galore_path (Path): Path to directory where trimmed reads will be stored.
        trim_galore_kwargs (Dict[str, Any]): A dictionary of Trim Galore command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--quality": "20", "--stringency": "3"}
        pattern (str): File name pattern that the FASTQ files to be processed should follow.
            Supports glob syntax relative to fastq_path.
        slurm_kwargs (Optional[Dict[str, Any]]): A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Trimmed FASTQ files are written to the specified output directory
        according to Trim Galore's naming conventions.

    Examples:
        >>> run_trim_galore(
        ...     fastq_path=Path("/data/raw_fastq"),
        ...     trim_galore_path=Path("/data/trimmed_fastq"),
        ...     trim_galore_kwargs={"--quality": "20", "--stringency": "3"},
        ...     slurm_kwargs={"--mem": "16G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run Cutadapt for each sample
    logging.info("Submitting SLURM jobs: ")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = trim_galore_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "trim_galore",
                "--paired",
                *list(chain(*trim_galore_kwargs.items())),
                "-o",
                sample_out_path,
                *sample_reads[:2],
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.trim_galore.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.trim_galore.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_multiqc(
    multiqc_path: Path,
    analyses_paths: Iterable[Path],
    multiqc_kwargs: Dict[str, Any],
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate MultiQC reports for multiple bioinformatics analyses.

    MultiQC is a tool that aggregates results from bioinformatics analyses across many
    samples into a single report. This function takes the paths to various analysis
    outputs (such as FastQC, alignment metrics, etc.) and generates a comprehensive
    quality control report.

    Args:
        multiqc_path: Path to directory where MultiQC reports will be stored.
        analyses_paths: List of paths containing analysis results to be included in the
            report. Each path should point to a directory containing output files from
            bioinformatics tools that MultiQC can parse.
        multiqc_kwargs: A dictionary of MultiQC command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--filename": "project_report", "--title": "My Project"}
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, the
            command will be run locally. Keys should include SLURM parameters like
            "--mem", "--cpus-per-task", etc.

    Returns:
        None. MultiQC report files are written to the specified output directory.

    Examples:
        >>> run_multiqc(
        ...     multiqc_path=Path("/data/multiqc_reports"),
        ...     analyses_paths=[Path("/data/fastqc_results"), Path("/data/star_results")],
        ...     multiqc_kwargs={"--title": "RNA-Seq QC Report", "--filename": "rnaseq_report"},
        ...     slurm_kwargs={"--mem": "8G", "--cpus-per-task": "2"}
        ... )
    """
    multiqc_path.mkdir(exist_ok=True, parents=True)

    # 1. Build command
    cmd_args = map(
        str,
        [
            "multiqc",
            *list(chain(*multiqc_kwargs.items())),
            "-o",
            multiqc_path,
            *analyses_paths,
        ],
    )

    # 2. Run command
    if slurm_kwargs is not None:
        slurm_kwargs["--error"] = str(multiqc_path.joinpath("multiqc.error.log"))
        slurm_kwargs["--output"] = str(multiqc_path.joinpath("multiqc.output.log"))

        log_file = multiqc_path.joinpath("multiqc.sbatch.log")
        SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=multiqc_path).submit(
            " ".join(cmd_args), "multi_qc", log_file
        )
    else:
        log_file = multiqc_path.joinpath("multiqc.log")
        run_cmd(cmd=cmd_args, log_path=log_file)


class RunMode(str, Enum):
    create_index = "create_index"
    use_index = "use_index"


def run_star(
    fastq_path: Path,
    star_path: Path,
    genome_path: Path,
    star_kwargs: Dict[str, Any],
    genome_fasta_files: Iterable[Path] = None,
    gtf_file: Path = None,
    run_mode: RunMode = RunMode.use_index,
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run STAR RNA-seq aligner on FASTQ files.

    STAR (Spliced Transcripts Alignment to a Reference) is a fast RNA-seq read mapper.
    This function supports both genome index generation and read alignment modes.
    When aligning reads, it processes multiple FASTQ files in parallel using either
    direct execution or SLURM job submission.

    Expected naming scheme of FASTQ files is: {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing FASTQ files.
        star_path: Path to directory where STAR alignment results will be stored.
        genome_path: Path to directory containing STAR genome index, or where
            the index will be generated if using create_index mode.
        star_kwargs: A dictionary of STAR command-line options. Keys will be passed
            as flags and values as arguments to those flags.
            Example: {"--outSAMtype": "BAM SortedByCoordinate", "--runThreadN": "8"}
        genome_fasta_files: Paths to the genome FASTA files. Required only when
            run_mode is set to create_index.
        gtf_file: Path to GTF/GFF file with gene annotations. Required only when
            run_mode is set to create_index.
        run_mode: Operation mode, either 'create_index' to generate STAR genome
            index or 'use_index' to align reads to an existing index.
        pattern: File name pattern that the FASTQ files to be processed should follow.
            Supports glob syntax relative to fastq_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Results are written to the specified output directories. In alignment mode,
        each sample produces its own set of output files in the star_path directory.

    Examples:
        >>> # First create the genome index
        >>> run_star(
        ...     fastq_path=None,
        ...     star_path=None,
        ...     genome_path=Path("/data/star_index"),
        ...     genome_fasta_files=[Path("/data/genome.fa")],
        ...     gtf_file=Path("/data/annotations.gtf"),
        ...     star_kwargs={"--sjdbOverhang": "100"},
        ...     run_mode=RunMode.create_index,
        ...     slurm_kwargs={"--mem": "32G", "--cpus-per-task": "8"}
        ... )
        >>>
        >>> # Then align reads using the created index
        >>> run_star(
        ...     fastq_path=Path("/data/raw_fastq"),
        ...     star_path=Path("/data/star_alignments"),
        ...     genome_path=Path("/data/star_index"),
        ...     star_kwargs={"--outSAMtype": "BAM SortedByCoordinate",
        ...                 "--runThreadN": "8"},
        ...     slurm_kwargs={"--mem": "32G", "--cpus-per-task": "8"}
        ... )
    """
    # 1. Setup
    genome_path.mkdir(exist_ok=True, parents=True)
    assert len(list(genome_path.glob("*"))) != 0 or run_mode == RunMode.create_index, (
        f"{genome_path} is empty. Either select a valid path containing "
        "genome indices or generate them by running the command again "
        'using the "--run-mode create_index"'
    )

    # 2. Create index if requested
    if run_mode == RunMode.create_index:
        assert gtf_file is not None and genome_fasta_files is not None, (
            "gtf_file and genome_fasta_files have to be provided when "
            "creating a genome index."
        )

        # 2.1. Build command
        cmd_args = map(
            str,
            [
                "STAR",
                "--runMode",
                "genomeGenerate",
                "--genomeDir",
                genome_path,
                "--genomeFastaFiles",
                *genome_fasta_files,
                "--sjdbGTFfile",
                gtf_file,
                *list(chain(*star_kwargs.items())),
            ],
        )

        # 2.2. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(genome_path.joinpath("star_genome.error.log"))
            slurm_kwargs["--output"] = str(
                genome_path.joinpath("star_genome.star.output.log")
            )

            log_file = genome_path.joinpath("create_index.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=genome_path).submit(
                " ".join(cmd_args), "create_index", log_file
            )
            logging.info("Submitted SLURM job to create genome index.")
        else:
            log_file = genome_path.joinpath("create_index.log")
            run_cmd(cmd=cmd_args, log_path=log_file)

        return

    # 3. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 4. Run star for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 4.1. Create intermediary paths
        sample_out_path = star_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 4.2. Build command
        cmd_args = map(
            str,
            [
                "STAR",
                "--runMode",
                "alignReads",
                "--genomeDir",
                genome_path,
                "--readFilesIn",
                *sample_reads,
                "--outFileNamePrefix",
                str(sample_out_path) + "/",
                *list(chain(*star_kwargs.items())),
            ],
        )

        # 4.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.star.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.star.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_bowtie2(
    fastq_path: Path,
    bowtie2_path: Path,
    bt2_path: Path,
    bowtie2_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Align reads to a reference genome using Bowtie2.

    Bowtie2 is a fast and memory-efficient tool for aligning sequencing reads to
    reference genomes. This function processes multiple FASTQ files in parallel using
    either direct execution or SLURM job submission. After alignment, the resulting
    SAM files are converted to BAM format and sorted by name using samtools.

    Expected naming scheme of FASTQ files is: {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing FASTQ files.
        bowtie2_path: Path to directory where Bowtie2 alignment results will be stored.
        bt2_path: Path to the Bowtie2 index files (without file extensions).
        bowtie2_kwargs: A dictionary of Bowtie2 command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--very-sensitive": "", "--no-unal": ""}
        pattern: File name pattern that the FASTQ files to be processed should follow.
            Supports glob syntax relative to fastq_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Alignment files are written to the specified output directory with
        the naming formats {sample_id}.sam, {sample_id}.bam, and {sample_id}.name_sorted.bam.

    Examples:
        >>> run_bowtie2(
        ...     fastq_path=Path("/data/trimmed_fastq"),
        ...     bowtie2_path=Path("/data/bowtie2_alignments"),
        ...     bt2_path=Path("/data/reference/genome"),
        ...     bowtie2_kwargs={"--very-sensitive": "", "--no-unal": ""},
        ...     slurm_kwargs={"--mem": "32G", "--cpus-per-task": "8"}
        ... )
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run bowtie2 for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = bowtie2_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "bowtie2",
                "-x",
                bt2_path,
                "-1",
                sample_reads[0],
                "-2",
                sample_reads[1],
                "-S",
                sample_out_path.joinpath(f"{sample_id}.sam"),
                "--met-file",
                sample_out_path.joinpath(f"{sample_id}.metrics.log"),
                *list(chain(*bowtie2_kwargs.items())),
                "&&",
                "samtools",
                "view",
                "-h",
                "-S",
                "-b",
                "-o",
                sample_out_path.joinpath(f"{sample_id}.bam"),
                sample_out_path.joinpath(f"{sample_id}.sam"),
                "&&",
                "samtools",
                "sort",
                "-n",
                "-o",
                sample_out_path.joinpath(f"{sample_id}.name_sorted.bam"),
                sample_out_path.joinpath(f"{sample_id}.bam"),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.bowtie2.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.bowtie2.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_bowtie2_summary(
    bowtie2_path: Path,
    pattern: str = "**/*.bowtie2.error.log",
    col_suffix: str = "_hg19",
) -> None:
    """
    Summarize Bowtie2 mapping results for multiple samples.

    This function processes Bowtie2 error log files (which contain alignment statistics)
    to generate individual sample summaries and a combined report for all samples.
    The summary includes metrics like sequencing depth, mapped fragment counts, and
    alignment rates.

    Args:
        bowtie2_path: Path to directory containing Bowtie2 alignment results.
        pattern: File name pattern that the Bowtie2 error log files should follow.
            Supports glob syntax relative to bowtie2_path.
        col_suffix: Suffix to append to column names in the output summary tables,
            typically used to indicate the reference genome version (e.g., "_hg19").

    Returns:
        None. Results are written to individual CSV files (one per sample) and a
        combined Excel file with alignment statistics for all samples.

    Examples:
        >>> run_bowtie2_summary(
        ...     bowtie2_path=Path("/data/bowtie2_alignments"),
        ...     pattern="**/*.bowtie2.error.log",
        ...     col_suffix="_hg38"
        ... )
    """
    all_stats = dict()
    for bowtie2_report_file in bowtie2_path.glob(pattern):
        sample_id = bowtie2_report_file.parent.stem
        stats = [line.strip() for line in bowtie2_report_file.read_text().split("\n")][
            :-1
        ]

        stats_dict = {
            "sequencing_depth": int(stats[0].split(" ")[0]),
            f"mapped_frags{col_suffix}": int(stats[3].split(" ")[0])
            + int(stats[4].split(" ")[0]),
            f"alignment_rate{col_suffix}": stats[5].split(" ")[0],
        }
        pd.Series(stats_dict).rename("value").to_csv(
            bowtie2_report_file.parent.joinpath("alignment_summary.csv")
        )
        all_stats[sample_id] = stats_dict

    pd.DataFrame(all_stats).transpose().rename_axis("sample_id").sort_index().to_excel(
        bowtie2_path.joinpath(f"alignment_summary_all_samples{col_suffix}.xlsx")
    )


def run_bam_to_bed(
    bam_path: Path,
    bed_path: Path,
    genome_file_path: Path,
    pattern: str = "**/*.name_sorted.bam",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convert BAM files to BED and bedgraph formats for genome browser visualization.

    This function processes BAM files into various BED formats suitable for genome
    browsers and coverage analysis. It extracts paired-end reads, cleans them to
    include only valid pairs, converts to fragment BED format, and generates coverage
    bedgraph files. The process creates UCSC-compatible bedgraph files that can be
    directly loaded into genome browsers.

    Expected naming scheme of BAM files is: {sample_id}.name_sorted.bam

    Args:
        bam_path: Path to directory containing name-sorted BAM files.
        bed_path: Path to directory where converted BED and bedgraph files will be stored.
        genome_file_path: Path to genome file containing chromosome sizes, used by
            bedtools genomecov to generate proper coverage files.
        pattern: File name pattern that the BAM files to be processed should follow.
            Supports glob syntax relative to bam_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Multiple files are created in the specified output directory for each
        sample, including:
        - {sample_id}.bed: Raw BEDPE output
        - {sample_id}.clean.bed: Filtered BEDPE file with valid pairs
        - {sample_id}.fragments.bed: BED3 format with fragment coordinates
        - {sample_id}.fragments.clean.bed: BED3 with invalid entries removed
        - {sample_id}.fragments.clean.bedraph: Coverage bedgraph
        - {sample_id}.fragments.clean.ucsc.bedraph: UCSC-compatible bedgraph

    Examples:
        >>> run_bam_to_bed(
        ...     bam_path=Path("/data/bowtie2_alignments"),
        ...     bed_path=Path("/data/bed_files"),
        ...     genome_file_path=Path("/data/reference/hg38.genome"),
        ...     slurm_kwargs={"--mem": "16G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all bam files per sample
    bam_files = dict()
    for bam_file in sorted(bam_path.glob(pattern)):
        sample_id = bam_file.stem.partition(".")[0]
        bam_files[sample_id] = bam_file

    # 2. Run for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(bam_files.items()):
        # 2.1. Create intermediary paths
        sample_out_path = bed_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "bedtools",
                "bamtobed",
                "-bedpe",
                "-i",
                bam_file,
                ">",
                sample_out_path.joinpath(f"{sample_id}.bed"),
                "&&",
                "awk",
                "'$1==$4 && $6-$2 < 1000 {print $0}'",
                sample_out_path.joinpath(f"{sample_id}.bed"),
                ">",
                sample_out_path.joinpath(f"{sample_id}.clean.bed"),
                "&&",
                "cut",
                "-f",
                "1,2,6",
                sample_out_path.joinpath(f"{sample_id}.clean.bed"),
                "|",
                "sort",
                "-k1,1",
                "-k2,2n",
                "-k3,3n",
                ">",
                sample_out_path.joinpath(f"{sample_id}.fragments.bed"),
                "&&",
                "awk",
                "'!/^.\t-1\t-1/'",
                sample_out_path.joinpath(f"{sample_id}.fragments.bed"),
                ">",
                sample_out_path.joinpath(f"{sample_id}.fragments.clean.bed"),
                "&&",
                "bedtools",
                "genomecov",
                "-bg",
                "-i",
                sample_out_path.joinpath(f"{sample_id}.fragments.clean.bed"),
                "-g",
                genome_file_path,
                ">",
                sample_out_path.joinpath(f"{sample_id}.fragments.clean.bedraph"),
                "&&",
                "awk",
                '{\'print "chr"$1,"\\t", $2, "\\t", $3, "\\t", $4\'}',
                sample_out_path.joinpath(f"{sample_id}.fragments.clean.bedraph"),
                ">",
                sample_out_path.joinpath(f"{sample_id}.fragments.clean.ucsc.bedraph"),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.bam_to_bed.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.bam_to_bed.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_bam_to_fastq(
    bam_path: Path,
    fastq_path: Path,
    bamToFastq_kwargs: Dict[str, Any],
    slurm_kwargs: Dict[str, Any],
    paired_end: bool = False,
    pattern: str = "**/*.bam",
) -> None:
    """
    Convert BAM files to FASTQ format using bamToFastq.

    This function uses bamToFastq (from the BEDTools suite) to extract sequences from
    BAM alignment files and convert them back to FASTQ format. After extraction, the
    resulting FASTQ files are compressed using pigz. The function processes multiple
    BAM files in parallel using direct execution or SLURM job submission.

    Expected naming scheme of BAM files is: {sample_id}.bam

    Args:
        bam_path: Path to directory containing BAM files.
        fastq_path: Path to directory where converted FASTQ files will be stored.
        bamToFastq_kwargs: A dictionary of bamToFastq command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--maxRecordLimitPerRG": "1000000"}
        slurm_kwargs: A dictionary of SLURM cluster batch job options. Keys should
            include SLURM parameters like "--mem", "--cpus-per-task", etc.
        paired_end: Boolean flag indicating if the sequences are paired-end reads.
            If True, will create separate FASTQ files for read 1 and read 2.
        pattern: File name pattern that the BAM files to be processed should follow.
            Supports glob syntax relative to bam_path.

    Returns:
        None. Converted FASTQ files are written to the specified output directory in
        gzip-compressed format with naming {sample_id}.{pair_n}.fastq.gz.

    Examples:
        >>> run_bam_to_fastq(
        ...     bam_path=Path("/data/aligned_bam"),
        ...     fastq_path=Path("/data/extracted_fastq"),
        ...     bamToFastq_kwargs={"--maxRecordLimitPerRG": "1000000"},
        ...     paired_end=True,
        ...     slurm_kwargs={"--mem": "16G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all bam files
    bam_files = dict()
    for bam_file in sorted(bam_path.glob(pattern)):
        sample_id = bam_file.stem.partition(".")[0]
        bam_files[sample_id] = bam_file

    # 2. Run bamToFastq for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(bam_files.items()):
        # 2.1. Create intermediary paths
        sample_out_path = fastq_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        if paired_end:
            read_1_fastq = sample_out_path.joinpath(f"{sample_id}.1.fastq")
            read_2_fastq = sample_out_path.joinpath(f"{sample_id}.2.fastq")
            cmd_args = map(
                str,
                [
                    "bedtools",
                    "bamtofastq",
                    "-i",
                    bam_file,
                    "-fq",
                    read_1_fastq,
                    "-fq2",
                    read_2_fastq,
                    *list(chain(*bamToFastq_kwargs.items())),
                    "&&",
                    "pigz",
                    "-p",
                    "4",
                    read_1_fastq,
                    read_2_fastq,
                ],
            )
        else:
            read_fastq = sample_out_path.joinpath(f"{sample_id}.fastq")
            cmd_args = map(
                str,
                [
                    "bedtools",
                    "bamtofastq",
                    "-i",
                    bam_file,
                    "-fq",
                    read_fastq,
                    *list(chain(*bamToFastq_kwargs.items())),
                    "&&",
                    "pigz",
                    "-p",
                    "4",
                    read_fastq,
                ],
            )

        # 2.3. Run command
        slurm_kwargs["--error"] = str(
            sample_out_path.joinpath(f"{sample_id}.bamtofastq.error.log")
        )
        slurm_kwargs["--output"] = str(
            sample_out_path.joinpath(f"{sample_id}.bamtofastq.output.log")
        )

        log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
        SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
            " ".join(cmd_args), sample_id, log_file
        )


def run_htseq_count(
    bam_path: Path,
    htseq_path: Path,
    gtf_file: Path,
    htseq_kwargs: Dict[str, Any],
    pattern: str = "**/*.bam",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run htseq-count to count reads mapping to genomic features.

    HTSeq-count is a tool that counts the number of reads mapping to each genomic
    feature in a BAM file, using a GTF/GFF annotation file. This function processes
    multiple BAM files in parallel (one job per sample) using either direct execution
    or SLURM job submission.

    Expected naming scheme of BAM files is assumed by parent directory name (sample_id).

    Args:
        bam_path: Path to directory containing BAM files.
        htseq_path: Path to directory where HTSeq count results will be stored.
        gtf_file: Path to the GTF/GFF file with gene annotations.
        htseq_kwargs: A dictionary of HTSeq-count command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--stranded": "yes", "--mode": "intersection-strict"}
        pattern: File name pattern that the BAM files to be processed should follow.
            Supports glob syntax relative to bam_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Count files are written to the specified output directory with
        the naming format {sample_id}.tsv.

    Examples:
        >>> run_htseq_count(
        ...     bam_path=Path("/data/aligned_bam"),
        ...     htseq_path=Path("/data/htseq_counts"),
        ...     gtf_file=Path("/data/annotations.gtf"),
        ...     htseq_kwargs={"--stranded": "yes", "--mode": "intersection-strict"},
        ...     slurm_kwargs={"--mem": "16G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all bam files
    bam_files = dict()
    for bam_file in sorted(bam_path.glob(pattern)):
        sample_id = bam_file.parent.name
        bam_files[sample_id] = bam_file

    # 2. Run htseq-counts for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(bam_files.items()):
        # 2.1. Create intermediary paths
        sample_out_path = htseq_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "htseq-count",
                bam_file,
                gtf_file,
                "-o",
                sample_out_path.joinpath(f"{sample_id}.htseq"),
                *list(chain(*htseq_kwargs.items())),
                ">",
                sample_out_path.joinpath(f"{sample_id}.tsv"),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.htseq.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.htseq.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_samtools_index_bam(
    bam_path: Path,
    samtools_kwargs: Dict[str, Any],
    pattern: str = "**/*.bam",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Index BAM files with samtools to enable random access.

    This function sorts and indexes BAM files using samtools, which is necessary for
    random access and visualization in genome browsers. The function processes multiple
    BAM files in parallel using either direct execution or SLURM job submission.

    Expected naming scheme of BAM files is: {sample_id}.bam

    Args:
        bam_path: Path to directory containing BAM files.
        samtools_kwargs: A dictionary of samtools command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--threads": "4", "--verbosity": "3"}
        pattern: File name pattern that the BAM files to be processed should follow.
            Supports glob syntax relative to bam_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Sorted BAM files and their indexes are written to the same directory as
        the original BAM files, with naming format {sample_id}.coord_sorted.bam and
        {sample_id}.coord_sorted.bam.bai.

    Examples:
        >>> run_samtools_index_bam(
        ...     bam_path=Path("/data/aligned_bam"),
        ...     samtools_kwargs={"--threads": "4"},
        ...     slurm_kwargs={"--mem": "8G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all bam files
    bam_files = dict()
    for bam_file in sorted(bam_path.glob(pattern)):
        sample_id = bam_file.stem.partition(".")[0]
        bam_files[sample_id] = bam_file

    # 2. Run samtools for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(bam_files.items()):
        # 2.1. Create intermediary paths
        sample_out_path = bam_file.parent
        sample_out_path.mkdir(parents=True, exist_ok=True)
        coord_sorted_bam = sample_out_path.joinpath(f"{sample_id}.coord_sorted.bam")

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "samtools",
                "sort",
                *list(chain(*samtools_kwargs.items())),
                "-o",
                coord_sorted_bam,
                bam_file,
                "&&",
                "samtools",
                "index",
                coord_sorted_bam,
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.samtools_index.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.samtools_index.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_fasterq_dump(
    sra_path: Path,
    ngc_filepath: Path,
    fastq_path: Path,
    fasterq_dump_kwargs: Dict[str, Any],
    pigz_kwargs: Dict[str, Any],
    pattern: str = "**/*.sra",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Extract FASTQ files from SRA format using fasterq-dump.

    This function processes multiple SRA files in parallel using either direct execution
    or SLURM job submission. After extraction, the resulting FASTQ files are compressed
    using pigz. The function also renames files to ensure consistent naming convention.

    Expected naming scheme of SRA files is: {sample_id}.sra

    Args:
        sra_path: Path to directory containing SRA files.
        ngc_filepath: Path to NGC file needed for decompression of protected datasets.
        fastq_path: Path to directory where extracted FASTQ files will be stored.
        fasterq_dump_kwargs: A dictionary of fasterq-dump command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--split-files": "", "--threads": "4"}
        pigz_kwargs: A dictionary of pigz command-line options for compressing the
            resulting FASTQ files. Example: {"-p": "4", "-9": ""}
        pattern: File name pattern that the SRA files to be processed should follow.
            Supports glob syntax relative to sra_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Extracted FASTQ files are written to the specified output directory in
        gzip-compressed format with naming convention {sample_id}.{pair_n}.fastq.gz.

    Examples:
        >>> run_fasterq_dump(
        ...     sra_path=Path("/data/raw_sra"),
        ...     ngc_filepath=Path("/data/credentials/access.ngc"),
        ...     fastq_path=Path("/data/extracted_fastq"),
        ...     fasterq_dump_kwargs={"--split-files": "", "--threads": "4"},
        ...     pigz_kwargs={"-p": "4", "-9": ""},
        ...     slurm_kwargs={"--mem": "16G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all sra files
    sra_files = dict()
    for sra_file in sorted(sra_path.glob(pattern)):
        sample_id = sra_file.stem.partition(".")[0]
        sra_files[sample_id] = sra_file

    # 2. Run fasterq-dump for each SRA file
    logging.info("Submitting SLURM jobs:")
    for sample_id, sra_file in tqdm(sra_files.items()):
        # 2.1. Create intermediary paths
        sample_out_path = fastq_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "fasterq-dump",
                sra_file,
                "--outdir",
                sample_out_path,
                "--ngc",
                ngc_filepath,
                *list(chain(*fasterq_dump_kwargs.items())),
                "&&",
                "cd",
                sample_out_path,
                "&&",
                "for file in *.fastq; do",
                "pigz",
                *list(chain(*pigz_kwargs.items())),
                "$file;",
                "done",
                "&&",
                "ls",
                "*.fastq.gz",
                "|",
                r"sed -r 's/(.*)_([0-9]+).fastq.gz/mv &" r" \\1.\\2.fastq.gz/g'",
                "|",
                "bash",
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.fasterq_dump.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.fasterq_dump.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_bismark_genome(
    genome_path: Path,
    bismark_genome_kwargs: Dict[str, Any],
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Prepare a genome for use with Bismark methylation analysis.

    Bismark genome preparation converts genome FASTA files into a format suitable for
    bisulfite sequence alignment. This involves creating bisulfite-converted versions
    of the genome and building Bowtie2 indexes for these converted sequences.

    Args:
        genome_path: Path to directory containing genome FASTA files to be converted.
        bismark_genome_kwargs: A dictionary of Bismark genome preparation options.
            Keys will be passed as flags and values as arguments to those flags.
            Example: {"--parallel": "4", "--bowtie2": ""}
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, the
            command will be run locally. Keys should include SLURM parameters like
            "--mem", "--cpus-per-task", etc.

    Returns:
        None. The Bismark genome index files are created in the specified genome_path.

    Examples:
        >>> run_bismark_genome(
        ...     genome_path=Path("/data/reference/hg38"),
        ...     bismark_genome_kwargs={"--bowtie2": "", "--parallel": "4"},
        ...     slurm_kwargs={"--mem": "32G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Create intermediary paths
    genome_path.mkdir(exist_ok=True, parents=True)

    # 2. Build command
    cmd_args = map(
        str,
        [
            "bismark_genome_preparation",
            *list(chain(*bismark_genome_kwargs.items())),
            genome_path,
        ],
    )

    # 3. Run command
    if slurm_kwargs is not None:
        slurm_kwargs["--error"] = str(genome_path.joinpath("bismark_genome.error.log"))
        slurm_kwargs["--output"] = str(
            genome_path.joinpath("bismark_genome.output.log")
        )

        log_file = genome_path.joinpath("bismark_genome.sbatch.log")
        SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=genome_path).submit(
            " ".join(cmd_args), "bismark_genome", log_file
        )
    else:
        log_file = genome_path.joinpath("bismark_genome.log")
        run_cmd(cmd=cmd_args, log_path=log_file)


def run_bismark_mapping(
    fastq_path: Path,
    genome_path: Path,
    bismark_path: Path,
    bismark_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Map bisulfite-treated reads to a reference genome using Bismark.

    Bismark is a tool for aligning bisulfite-converted reads to a reference genome
    and determining cytosine methylation states. This function processes multiple FASTQ
    files in parallel using direct execution or SLURM job submission.

    Expected naming scheme of FASTQ files is: {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing FASTQ files.
        genome_path: Path to Bismark-converted genome reference directory.
        bismark_path: Path to directory where Bismark mapping results will be stored.
        bismark_kwargs: A dictionary of Bismark command-line options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--bowtie2": "", "--multicore": "4"}
        pattern: File name pattern that the FASTQ files to be processed should follow.
            Supports glob syntax relative to fastq_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Alignment files are written to the specified output directory with
        the naming format {sample_id}.bam.

    Examples:
        >>> run_bismark_mapping(
        ...     fastq_path=Path("/data/trimmed_fastq"),
        ...     genome_path=Path("/data/reference/bismark_hg38"),
        ...     bismark_path=Path("/data/bismark_alignments"),
        ...     bismark_kwargs={"--bowtie2": "", "--multicore": "4"},
        ...     slurm_kwargs={"--mem": "32G", "--cpus-per-task": "8"}
        ... )
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run bismark for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = bismark_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "bismark",
                "--output_dir",
                str(sample_out_path),
                "--genome",
                str(genome_path),
                *list(chain(*bismark_kwargs.items())),
                *sample_reads[:2],
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_bismark_meth_extract(
    genome_path: Path,
    bismark_path: Path,
    bismark_kwargs: Dict[str, Any],
    pattern: str = "**/*.bam",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Extract methylation information from Bismark alignment files.

    Bismark methylation extractor processes Bismark alignment files (.bam) to determine
    the methylation state of cytosines. It generates methylation call files and reports
    that can be used for downstream analysis. This function also runs bismark2report
    to generate HTML reports summarizing the methylation extraction results. The function
    processes multiple BAM files in parallel using direct execution or SLURM job submission.

    Expected naming scheme of BAM files is: {sample_id}.bam

    Args:
        genome_path: Path to directory containing reference genome used for the alignment.
        bismark_path: Path to directory containing Bismark alignment files (.bam) and
            where methylation extraction results will be stored.
        bismark_kwargs: A dictionary of Bismark methylation extractor options. Keys will be
            passed as flags and values as arguments to those flags.
            Example: {"--comprehensive": "", "--bedGraph": "", "--buffer_size": "10G"}
        pattern: File name pattern that the BAM files to be processed should follow.
            Supports glob syntax relative to bismark_path.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If None, each
            command will be run locally in sequential order. Keys should include SLURM
            parameters like "--mem", "--cpus-per-task", etc.

    Returns:
        None. Methylation extraction results are written to the specified output directory
        including cytosine reports, M-bias plots, methylation coverage files, and summary
        HTML reports.

    Examples:
        >>> run_bismark_meth_extract(
        ...     genome_path=Path("/data/reference/bismark_hg38"),
        ...     bismark_path=Path("/data/bismark_alignments"),
        ...     bismark_kwargs={"--comprehensive": "", "--bedGraph": "",
        ...                     "--buffer_size": "10G", "--parallel": "4"},
        ...     slurm_kwargs={"--mem": "32G", "--cpus-per-task": "4"}
        ... )
    """
    # 1. Get all bam files
    bam_files = dict()
    for bam_file in sorted(bismark_path.glob(pattern)):
        sample_id = bam_file.stem.partition(".")[0]
        bam_files[sample_id] = bam_file

    # 2. Run bismark methylation extractor for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(bam_files.items()):
        # 2.1. Create intermediary paths
        sample_out_path = bam_file.parent
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "bismark_methylation_extractor",
                "--output",
                str(sample_out_path),
                "--gzip",
                *list(chain(*bismark_kwargs.items())),
                bam_file,
                "&&",
                "cd",
                str(sample_out_path),
                "&&",
                "bismark2report",
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark_extract.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark_extract.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)
