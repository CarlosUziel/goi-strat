"""
This script spawns multiple slurm jobs, each processing one sample. This allows
faster processing of a whole batch of samples, since each sample is run in
parallel (depending on the cluster resources).
"""

import logging
from collections import defaultdict
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from slurm.slurm_job_submitter import SlurmJobSubmitter
from tqdm.rich import tqdm

from data.utils import run_cmd


def run_fastqc(
    fastq_path: Path,
    fastqc_path: Path,
    fastqc_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run fastqc on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing fastq files.
        fastqc_path: Path to trimmed reads.
        fastqc_kwargs: A dictionary of FastQC options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
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
    Run cutadapt on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing fastq files.
        cutadapt_path: Path to store trimmed reads.
        fwd_adapter_file: File containing forward adapter sequences.
        rv_adapter_file: File containing reverse adapter sequences.
        cutadapt_kwargs: A dictionary of cutadapt options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
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
    Run trim galore on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing fastq files.
        cutadapt_path: Path to store trimmed reads.
        cutadapt_kwargs: A dictionary of cutadapt options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
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
    Generate multiqc reports based on the results paths given.

    Args:
        multiqc_path: Path to store multiqc reports.
        analyses_paths: List of paths to obtain analysis results from.
        multiqc_kwargs: A dictionary of MULTIQC options.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
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
    Run star on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Genome files can be obtained from the ENSEMBL ftp server. For example, Homo
    sapiens files can be obtained from:
        https://www.ensembl.org/Homo_sapiens/Info/Index

    Args:
        fastq_path: Path to directory containing fastq files.
        star_path: Path to store aligned reads.
        genome_path: Location of genome files, generated by star in a previous
            step.
        star_kwargs: A list of star options. Option keys followed by
            their values (e.g. [--an-option, value]).
        genome_fasta_files: Paths to the fasta files with the genome sequences.
        gtf_file: path to the GTF file with annotations.
        run_mode: Whether to run star for alignment of reads or to create a
            genome index (necessary for alignment).
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
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


def run_htseq_count(
    bam_path: Path,
    htseq_path: Path,
    gtf_file: Path,
    htseq_kwargs: Dict[str, Any],
    pattern: str = "**/*.bam",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run htseq-count on bam files.

    Args:
        bam_path: Path to directory containing fastq files.
        htseq_path: Path to store htseq files.
        gtf_file: path to the GTF file with annotations.
        htseq_kwargs: A list of htseq-counts options. Option keys followed by
            their values (e.g. [--an-option, value]).
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all bam files
    samples_bams = {}
    for bam_file in sorted(bam_path.glob(pattern)):
        sample_id = bam_file.parent.name
        samples_bams[sample_id] = bam_file

    # 2. Run fastqc for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(samples_bams.items()):
        # 2.1. Create intermediary paths
        sample_out_path = htseq_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)
        htseq_out = sample_out_path.joinpath(f"{sample_id}.tsv")

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "htseq-count",
                *list(chain(*htseq_kwargs.items())),
                "-c",
                htseq_out,
                bam_file,
                gtf_file,
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.htseq_count.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.htseq_count.output.log")
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
    Index .bam files with samtools

    Args:
        bam_path: Path to directory containing fastq files.
        samtools_path: Path to store htseq files.
        samtools_kwargs: A dictionary of htseq-counts options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all bam files
    samples_bams = {}
    for bam_file in sorted(bam_path.glob(pattern)):
        if len(bam_file.name.split(".")) < 3:
            sample_id = bam_file.parent.name
            samples_bams[sample_id] = bam_file

    # 2. Run fastqc for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(samples_bams.items()):
        # 2.1. Create intermediary paths
        bam_sorted_file = bam_file.parent.joinpath(f"{sample_id}.coord_sorted.bam")
        bam_index_file = bam_sorted_file.with_suffix(".bai")

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "samtools",
                "sort",
                *list(chain(*samtools_kwargs.items())),
                "-o",
                bam_sorted_file,
                bam_file,
                "&&",
                "samtools",
                "index",
                *list(chain(*samtools_kwargs.items())),
                "-o",
                bam_index_file,
                bam_sorted_file,
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                bam_file.parent.joinpath(f"{sample_id}.samtools_index.error.log")
            )
            slurm_kwargs["--output"] = str(
                bam_file.parent.joinpath(f"{sample_id}.samtools_index.output.log")
            )

            log_file = bam_file.parent.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=bam_file.parent).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = bam_file.parent.joinpath(f"{sample_id}.log")
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
    Run bamToFastq on bam files. Expected naming scheme of bam files is:
        {sample_id}.bam

    Args:
        bam_path: Path to directory containing fastq files.
        fastq_path: Path to trimmed reads.
        bamToFastq_kwargs: A dictionary of bamToFastq options.
        paired_end: Whether reads are paired-end.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all fastq files per sample
    bam_files = defaultdict(list)
    for bam_file in sorted(bam_path.glob(pattern)):
        sample_id = bam_file.stem.partition(".")[0]
        bam_files[sample_id] = bam_file

    # 2. Run bamToFastq for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(bam_files.items()):
        # 2.1. Create intermediary paths
        sample_out_path = fastq_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        read_1 = sample_out_path.joinpath(f"{sample_id}.1.fastq")

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "bamToFastq",
                "-i",
                bam_file,
                "-fq",
                read_1,
                *(
                    ["-fq2", sample_out_path.joinpath(f"{sample_id}.2.fastq")]
                    if paired_end
                    else []
                ),
                *list(chain(*bamToFastq_kwargs.items())),
                "&&",
                "pigz",
                "-r",
                sample_out_path.joinpath("*.fastq"),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.bamToFastq.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.bamToFastq.output.log")
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
    Run fastq-dump on SRA files. Expected naming scheme of sra files is:
        {sample_id}.sra

    Args:
        sra_path: Path to directory containing sra files.
        ngc_filepath: Path to NGC file needed for decompression.
        fastq_path: Path to store raw fastq files.
        fasterq_dump_kwargs: A dictionary of fastq-dump options.
        pigz_kwargs: Options to compress .fastq files.
        pattern: File name pattern the sra files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all fastq files per sample
    samples_sra = dict()
    for sra_file in sorted(sra_path.glob(pattern)):
        samples_sra[sra_file.stem] = sra_file.parent

    # 2. Run fastq-dump for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, sample_sra_dir in tqdm(samples_sra.items()):
        # 2.1. Create intermediary paths
        sample_out_path = fastq_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "fasterq-dump",
                "--ngc",
                ngc_filepath,
                "--outdir",
                sample_out_path,
                *list(chain(*fasterq_dump_kwargs.items())),
                sample_sra_dir,
                # from .fastq to .fastq.gz
                "&&",
                "pigz",
                *list(chain(*pigz_kwargs.items())),
                f"{sample_out_path}/*.fastq",
                "&&",
                'for f in *.fastq.gz; do mv "$f" "${f//_/.}"; done',
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
    Run bismark_genome_preparation.

    Args:
        genome_path: Path to genome fasta files to be converted.
        bismark_genome_kwargs: A dictionary of Bismark Genome Preparation options.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Build command
    cmd_args = map(
        str,
        [
            "bismark_genome_preparation",
            *list(chain(*bismark_genome_kwargs.items())),
            genome_path,
        ],
    )

    # 2. Run command
    if slurm_kwargs is not None:
        slurm_kwargs["--error"] = str(genome_path.joinpath("bismark_genome.error.log"))
        slurm_kwargs["--output"] = str(
            genome_path.joinpath("bismark_genome.output.log")
        )

        log_file = genome_path.joinpath("bismark_genome_preparation.sbatch.log")
        SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=genome_path).submit(
            " ".join(cmd_args), 0, log_file
        )
    else:
        log_file = genome_path.joinpath("bismark_genome_preparation.log")
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
    Run bismark mapping on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing fastq files.
        genome_path: Path to genome fasta files to be converted.
        bismark_path: Path to store mapping results.
        bismark_kwargs: A dictionary of Bismark Genome Preparation options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run Bismark for each sample
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
                "-o",
                sample_out_path,
                *list(chain(*bismark_kwargs.items())),
                genome_path,
                "-1",
                sample_reads[0],
                "-2",
                sample_reads[1],
                "&&",
                "mv",
                f"{sample_out_path}/*.bam",
                f"{sample_out_path}/{sample_id}.bam",
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark_mapping.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark_mapping.output.log")
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
    Run bismark methylation extractor on mapped bismark .bam files. Expected naming
        scheme of bam files is: {sample_id}.bam

    Args:
        genome_path: Path to genome fasta files to be converted.
        bismark_path: Path to read .bam files and store methylation results.
        bismark_kwargs: A dictionary of Bismark Genome Preparation options.
        pattern: File name pattern the bam files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all fastq files per sample
    samples_bams = dict()
    for bam_file in sorted(bismark_path.glob(pattern)):
        sample_id = bam_file.stem.partition(".")[0]
        samples_bams[sample_id] = bam_file

    # 2. Run Bismark for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(samples_bams.items()):
        # 2.1. Create intermediary paths
        sample_out_path = bismark_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "bismark_methylation_extractor",
                "--genome",
                genome_path,
                "-o",
                sample_out_path,
                *list(chain(*bismark_kwargs.items())),
                bam_file,
                "&&",
                "bismark2report",
                "--alignment_report",
                sample_out_path.joinpath(f"{sample_id}*_bismark_bt2_PE_report.txt"),
                "--splitting_report",
                sample_out_path.joinpath(f"{sample_id}.splitting_report.txt"),
                "--mbias_report",
                sample_out_path.joinpath(f"{sample_id}.M-bias.txt"),
                "--nucleotide_report",
                sample_out_path.joinpath(
                    f"{sample_id}*_bismark_bt2_pe.nucleotide_stats.txt"
                ),
                "--dir",
                sample_out_path,
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark_meth_extract.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.bismark_meth_extract.output.log")
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
    Run bowtie2 on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Genome files can be obtained from the ENSEMBL ftp server. For example, Homo
    sapiens files can be obtained from:
        https://www.ensembl.org/Homo_sapiens/Info/Index

    Args:
        fastq_path: Path to directory containing fastq files.
        bowtie2_path: Path to store aligned reads.
        bt2_path: Location of bowtie2 index.
        bowtie2_kwargs: A list of star options. Option keys followed by
            their values (e.g. [--an-option, value]).
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
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
    Summarize bowtie2 mapping results for multiple samples.

    Args:
        bowtie2_path: Path to directory containing bowtie2 results.
        pattern: File name pattern the bowtie2 reports should follow.
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
    Convert to paired end BED files using bedtools bamtobed with the -bedpe flag,
    then selecting the 5' and 3' coordinates of the read pair to generate a new BED3
    file, and finally converting that file to a bedgraph using bedtools genomecov.

    Args:
        bam_path: Path to directory containing .bam files.
        bed_path: Path to store converted bedgraph files.
        genome_file_path: Path to genome fasta file.
        pattern: File name pattern the .bam files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
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
