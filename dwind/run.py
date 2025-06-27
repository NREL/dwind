"""Enables running dwind as a CLI tool, the primary interface for working with dwind."""

from __future__ import annotations

import sys
import tomllib
from enum import Enum
from typing import Annotated
from pathlib import Path

import typer
import pandas as pd


# from memory_profiler import profile


app = typer.Typer()

DWIND = Path("/projects/dwind/agents")


class Sector(str, Enum):
    """Typer helper for validating the sector input.

    - "fom": Front of meter
    - "btm": Behind the meter
    """

    fom = "fom"
    btm = "btm"


class Scenario(str, Enum):
    """Scenario to run.

    The only current option is "baseline".
    """

    baseline = "baseline"


def year_callback(ctx: typer.Context, param: typer.CallbackParam, value: int):
    """Typer helper to validate the year input.

    Parameters
    ----------
    ctx : typer.Context
        The Typer context.
    param : typer.CallbackParam
        The Typer parameter.
    value : int
        User input for the analysis year basis, must be one of 2022, 2024, or 2025.

    Returns:
    -------
    int
        The input :py:param:`value`, if it is a valid input.

    Raises:
    ------
    typer.BadParameter
        Raised if the input is not one of 2022, 2024, or 2025.
    """
    valid_years = (2022, 2024, 2025, 2035, 2040)
    if ctx.resilient_parsing:
        return
    if value not in valid_years:
        raise typer.BadParameter(f"Only {valid_years} are valid options for `year`, not {value}.")
    return value


def load_agents(
    file_name: str | Path | None = None,
    location: str | None = None,
    sector: str | None = None,
) -> pd.DataFrame:
    """Load the agent file based on a filename or the location and sector to a Pandas DataFrame,
    and return the data frame.

    Args:
        file_name (str | Path | None, optional): Name of the agent file, if not auto-generating from
            the :py:attr:`location` and :py:attr:`sector` inputs. Defaults to None.
        location (str | None, optional): The name of the location or grouping, such as
            "colorado_larimer" or "priority1". Defaults to None.
        sector (str | None, optional): The name of the section. Must be one of "btm" or "fom".
            Defaults to None.

    Returns:
        pd.DataFrame: The agent DataFrame.
    """
    from dwind.model import Agents

    if file_name is None and (location is None or sector is None):
        raise ValueError("One of `file_name` or `location` and `sector` must be provided.")

    f_agents = (
        file_name if file_name is not None else DWIND / f"{location}/agents_dwind_{sector}.parquet"
    )
    if not f_agents.exists():
        f_agents = f_agents.with_suffix(".pickle")
    agents = Agents(agent_file=f_agents).agents
    return agents


@app.command()
def run_hpc(
    location: Annotated[
        str, typer.Option(help="The state, state_county, or priority region to run.")
    ],
    sector: Annotated[
        Sector, typer.Option(help="One of fom (front of meter) or btm (back-of-the-meter).")
    ],
    scenario: Annotated[
        Scenario,
        typer.Option(help="The scenario to run (baseline is the current only option)."),
    ],
    year: Annotated[
        int,
        typer.Option(
            callback=year_callback,
            help="The assumption year for the analysis. Options are 2022, 2024, and 2025.",
        ),
    ],
    repository: Annotated[
        str, typer.Option(help="Path to the dwind repository to use when running the model.")
    ],
    nodes: Annotated[
        int,
        typer.Option(
            help="Number of HPC nodes or CPU nodes to run on. -1 indicates 75% of CPU limit."
        ),
    ],
    allocation: Annotated[str, typer.Option(help="HPC allocation name.")],
    memory: Annotated[int, typer.Option(help="Node memory, in GB (HPC only).")],
    walltime: Annotated[int, typer.Option(help="Node walltime request, in hours.")],
    feature: Annotated[
        str,
        typer.Option(
            help=(
                "Additional flags for the SLURM job, using formatting such as"
                " --qos=high or --depend=[state:job_id]."
            )
        ),
    ],
    env: Annotated[
        str,
        typer.Option(
            help="The path to the dwind Python environment that should be used to run the model."
        ),
    ],
    model_config: Annotated[
        str, typer.Option(help="Complete file name and path of the model configuration file")
    ],
    dir_out: Annotated[
        str, typer.Option(help="Path to where the chunked outputs should be saved.")
    ],
    stdout_path: Annotated[str | None, typer.Option(help="The path to write stdout logs.")] = None,
):
    """Run dwind via the HPC multiprocessing interface."""
    sys.path.append(repository)
    from dwind.mp import MultiProcess

    # NOTE: collect_by_priority has been removed but may need to be reinstated

    mp = MultiProcess(
        location=location,
        sector=sector,
        scenario=scenario,
        year=year,
        env=env,
        n_nodes=nodes,
        memory=memory,
        walltime=walltime,
        allocation=allocation,
        feature=feature,
        repository=repository,
        model_config=model_config,
        dir_out=dir_out,
        stdout_path=stdout_path,
    )

    agent_df = load_agents(location=location, sector=sector)
    mp.run_jobs(agent_df)


@app.command("run-config")
def run_hpc_from_config(
    config_path: Annotated[
        str, typer.Argument(help="Path to configuration TOML with run and model parameters.")
    ],
):
    """Run dwind via the HPC multiprocessing interface from a configuration file."""
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    print(config)

    run_hpc(**config)


@app.command()
def run_chunk(
    # start: Annotated[int, typer.Option(help="chunk start index")],
    # end: Annotated[int, typer.Option(help="chunk end index")],
    location: Annotated[
        str, typer.Option(help="The state, state_county, or priority region to run.")
    ],
    sector: Annotated[
        Sector, typer.Option(help="One of fom (front of meter) or btm (back-of-the-meter).")
    ],
    scenario: Annotated[str, typer.Option(help="The scenario to run, such as baseline.")],
    year: Annotated[
        int, typer.Option(callback=year_callback, help="The year basis of the scenario.")
    ],
    chunk_ix: Annotated[int, typer.Option(help="Chunk number/index. Used for logging.")],
    out_path: Annotated[str, typer.Option(help="save path")],
    repository: Annotated[
        str, typer.Option(help="Path to the dwind repository to use when running the model.")
    ],
    model_config: Annotated[
        str, typer.Option(help="Complete file name and path of the model configuration file")
    ],
):
    """Run a chunk of a dwind model. Internal only, do not run outside the context of a
    chunked analysis.
    """
    # Import the correct version of the library
    sys.path.append(repository)
    from dwind.model import Model

    agent_file = Path(out_path).resolve() / f"agents_{chunk_ix}.pqt"
    agents = load_agents(file_name=agent_file)
    agent_file.unlink()

    model = Model(
        agents=agents,
        location=location,
        sector=sector,
        scenario=scenario,
        year=year,
        chunk_ix=chunk_ix,
        out_path=out_path,
        model_config=model_config,
    )
    model.run()


@app.command()
def run(
    location: Annotated[
        str, typer.Option(help="The state, state_county, or priority region to run.")
    ],
    sector: Annotated[
        Sector, typer.Option(help="One of fom (front of meter) or btm (back-of-the-meter).")
    ],
    scenario: Annotated[str, typer.Option(help="The scenario to run, such as 'baseline'.")],
    year: Annotated[int, typer.Option(help="The year basis of the scenario.")],
    out_path: Annotated[str, typer.Option(help="save path")],
    repository: Annotated[
        str, typer.Option(help="Path to the dwind repository to use when running the model.")
    ],
    model_config: Annotated[
        str, typer.Option(help="Complete file name and path of the model configuration file")
    ],
):
    """Run the dwind model. Does not yet work, do not run unless dwind has been configured
    to not rely on Kestrel usage.
    """
    # Import the correct version of the library
    sys.path.append(repository)
    from dwind.model import Model

    agents = load_agents(location=location, sector=sector)
    model = Model(
        agents=agents,
        location=location,
        sector=sector,
        scenario=scenario,
        year=year,
        out_path=out_path,
        model_config=model_config,
    )
    model.run()


if __name__ == "__main__":
    app()
