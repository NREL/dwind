"""Provides shared utilities among the CLI subcommand modules."""

from __future__ import annotations

from pathlib import Path

import typer
import pandas as pd
from rich.style import Style
from rich.table import Table
from rich.console import Console

from dwind.config import Year, Configuration


console = Console()


def year_callback(ctx: typer.Context, param: typer.CallbackParam, value: int):
    """Typer helper to validate the year input.

    Args:
        ctx (typer.Context): The Typer context.
        param (typer.CallbackParam): The Typer parameter.
        value (int): User input for the analysis year basis, must be one of 2022, 2024, or 2025.

    Returns:
        int: The input :py:attr:`value`, if it is a valid input.

    Raises:
        typer.BadParameter: Raised if the input is not one of 2022, 2024, or 2025.
    """
    if ctx.resilient_parsing:
        return
    try:
        year = Year(value)
    except ValueError as e:
        raise typer.BadParameter(
            f"Only {Year.values()} are valid options for `year`, not {value}."
        ) from e
    return year


def load_agents(
    file_name: Path | None = None,
    location: str | None = None,
    sector: str | None = None,
    model_config: str | Path | None = None,
    *,
    prepare: bool = False,
) -> pd.DataFrame:
    """Load the agent file based on a filename or the location and sector to a Pandas DataFrame,
    and return the data frame.

    Args:
        file_name (Path | None, optional): Name of the agent file, if not auto-generating from
            the :py:attr:`location` and :py:attr:`sector` inputs. Defaults to None.
        location (str | None, optional): The name of the location or grouping, such as
            "colorado_larimer" or "priority1". Defaults to None.
        sector (str | None, optional): The name of the section. Must be one of "btm" or "fom".
            Defaults to None.
        model_config: (str | Path, optional): The file name and path for the overarching model
            configuration containing SQL and file references, scenario configurations, and etc.
            Defaults to None.
        prepare (bool, optional): True if loading pre-chunked and prepared agent data, which should
            bypass the standard column checking for additional joins, by default False.

    Returns:
        pd.DataFrame: The agent DataFrame.
    """
    from dwind.model import Agents

    if file_name is None and (location is None or sector is None):
        raise ValueError("One of `file_name` or `location` and `sector` must be provided.")

    config = Configuration(model_config)
    f_agents = (
        file_name
        if file_name is not None
        else config.project.DATA_DIR / f"{location}/agents_dwind_{sector}.parquet"
    )
    if not isinstance(f_agents, Path):
        f_agents = Path(f_agents).resolve()

    alternative_suffix = (".pqt", ".parquet", ".pkl", ".pickle", ".csv")
    base_style = Style.parse("cyan")
    if not f_agents.exists():
        for suffix in alternative_suffix:
            if (new_fn := f_agents.with_suffix(suffix)).exists():
                if new_fn != f_agents:
                    msg = (
                        f"Using alternative agent file: {new_fn}\n\t"
                        f"Requested agent file: {f_agents}"
                    )
                    console.print(msg, style=base_style)
                    f_agents = new_fn
                    break

    if prepare:
        return Agents.load_and_prepare_agents(
            agent_file=f_agents, sector=sector, model_config=model_config
        )
    return Agents.load_agents(agent_file=f_agents)


def print_status_table(
    summary: pd.DataFrame,
    id_col: str = "job",
    chunk_col: str | None = None,
    status_col: str = "status",
) -> None:
    """Prints a nicely formatted pandas DataFrame to the console.

    Args:
        summary (pd.DataFrame): Summary DataFrame of the job (:py:attr:`id_col`), chunk
            (:py:attr:`chunk_col`), and status (:py:attr:`status_col`).
        id_col (str): Name of the job ID column.
        chunk_col (str): Name of the chunk index column.
        status_col (str): Name of the final job status column.
    """
    table = Table()
    table.add_column("Job ID")
    if chunk_col is not None:
        table.add_column("Agent Chunk")
    table.add_column("Status")

    if chunk_col is None:
        cols = [id_col, status_col]
        secondary_sort = status_col
    else:
        cols = [id_col, chunk_col, status_col]
        secondary_sort = chunk_col
    summary = summary[cols].astype({c: str for c in cols})
    for _, row in summary.sort_values([status_col, secondary_sort]).iterrows():
        table.add_row(*row.tolist())

    console.print(table)


def cleanup_chunks(dir_out: str | None, which: str) -> None:
    """Deletes the temporary agent chunk files, typically in
    ``/path/to/analysis_scenario/chunk_files/agents``, and of the form ``agents_{ix}.pqt``.

    Args:
        dir_out (str | None): The same :py:attr:`dir_out` passed to the run command. The
            ``chunk_files/agent_chunks`` will be automatically added to the path.
        which (str): One of "agents" or "results" to indicate which chunked files should be deleted.
    """
    dir_out = Path.cwd() if dir_out is None else Path(dir_out).resolve()
    out_path = dir_out / "chunk_files"
    if which == "agents":
        out_path = out_path / "agent_chunks"
    elif which != "results":
        raise ValueError("`which` must be one 'agents' or 'results'.")

    for f in out_path.iterdir():
        if f.suffix == ".pqt":
            f.unlink()
            print(f"Removed: {f}")
