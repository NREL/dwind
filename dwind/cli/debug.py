"""Enables running dwind as a CLI tool, the primary interface for working with dwind."""

from __future__ import annotations

import itertools
from typing import Optional, Annotated
from pathlib import Path

import typer
import pandas as pd
from rich.console import Console

from dwind.cli import utils
from dwind.utils import hpc


# Create the main app with the primary "run", "debug", and "collect" commands
app = typer.Typer()

console = Console()


@app.command()
def job_summary(
    jobs: Annotated[list[str], typer.Argument(help="Job ID(s) to check for the final run status.")],
    chunks: Annotated[
        Optional[list[int]],  # noqa
        typer.Option(
            "--chunks",
            "-c",
            help="Corresponding chunk indexes for each job ID.",
        ),
    ] = None,
):
    """Print a table summary of the provided job id(s) and their final run status."""
    if chunks is not None:
        N = len(jobs)
        M = len(chunks)
        if N != M:
            raise ValueError(f"Length of inputs to 'jobs' ({N}) not equal to 'chunks' ({M})")

    jobs = hpc.get_finished_run_status(jobs)
    summary = (
        pd.DataFrame.from_dict(jobs, orient="index", columns=["status"])
        .reset_index()
        .rename(columns={"index": "job"})
    )
    if chunks is None:
        utils.print_status_table(summary)
    else:
        summary["chunk"] = chunks
        utils.print_status_table(summary, chunk_col="chunk")


@app.command()
def missing_agents_from_chunks(
    dir_out: Annotated[
        str,
        typer.Argument(
            help=(
                "Path to where the chunked outputs were saved. Should match the inputs to the"
                " run command."
            )
        ),
    ],
    chunks: Annotated[
        Optional[list[int]], typer.Option(help="If used, specify chunked indicies to compare.")  # noqa
    ] = None,
):
    """Compare the agent "gid" field between the chunked agents and results files, and save the
    missing chunk index and agent gid as a 1-column CSV file.
    """
    dir_out = Path(dir_out).resolve()
    results_path = dir_out / "chunk_files"
    agents_path = results_path / "agent_chunks"

    missing = []
    agent_files = {
        f.name.split("_")[-1]: f
        for f in agents_path.iterdir()
        if f.suffix == ".pqt" and f.name.startswith("agents")
    }
    results_files = {f.name.split("_")[-1]: f for f in results_path.iterdir() if f.suffix == ".pqt"}
    shared_chunks = list({*agent_files}.intersection([*results_files]))
    if chunks is not None:
        shared_chunks = [c for c in shared_chunks if c in chunks]

    for chunk in shared_chunks:
        agents = pd.read_parquet(agent_files[chunk])[["gid"]]
        results = pd.read_parquet(results_files[chunk])[["gid"]]
        missing_gid = set(agents.gid.values).difference(results.gid.values)
        missing.extend(list(zip(itertools.repeat(chunk), missing_gid)))

    if missing:
        df = pd.DataFrame(missing, columns=["chunk", "gid"])
        df.to_csv(dir_out / "missing_agents_by_chunk.csv", index=False)
        print(f"Saved missing agents to: {dir_out / 'missing_agents_by_chunk.csv'}")
    else:
        print("All agents produced results.")


@app.command()
def missing_agents(
    agents_file: Annotated[str, typer.Argument(help="Full file path and name of the agents file.")],
    results_file: Annotated[
        str, typer.Argument(help="Full file path and name of the results file.")
    ],
    dir_out: Annotated[
        str,
        typer.Argument(
            help="Full file path and name for where to save the list of missing agent gids."
        ),
    ],
):
    """Compare the agent "gid" field between the chunked agents and results files, and save the
    missing chunk index and agent gid as a 2-column CSV file.
    """
    results_file = Path(results_file).resolve()
    agents_file = Path(agents_file).resolve()
    dir_out = Path(dir_out).resolve()

    agents = pd.read_parquet(agents_file)[["gid"]]
    results = pd.read_parquet(results_file)[["gid"]]
    missing_gid = set(agents.gid.values).difference(results.gid.values)

    if missing_gid:
        df = pd.DataFrame(missing_gid, columns=["gid"])
        df.to_csv(dir_out / "missing_agents.csv", index=False)
        print(f"Saved missing agents to: {dir_out / 'missing_agents.csv'}")
    else:
        print("All agents produced results.")


if __name__ == "__main__":
    app()
