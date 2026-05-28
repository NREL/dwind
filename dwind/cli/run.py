"""Enables running dwind as a CLI tool, the primary interface for working with dwind."""

from __future__ import annotations

import sys
from typing import Optional, Annotated
from pathlib import Path

import typer
from rich.pretty import pprint

from dwind.cli import debug, utils, collect
from dwind.model import Model
from dwind.config import Sector, Scenario, IncentiveScenario


# fmt: off
if sys.version_info >= (3, 11):  # noqa
    import tomllib
else:
    import tomli as tomllib
# fmt: on

app = typer.Typer()


@app.command()
def hpc(
    location: Annotated[
        str, typer.Argument(help="The state, state_county, or priority region to run.")
    ],
    sector: Annotated[
        Sector, typer.Argument(help="One of fom (front of meter) or btm (back-of-the-meter).")
    ],
    scenario: Annotated[
        Scenario,
        typer.Argument(help="The scenario to run (baseline is the current only option)."),
    ],
    incentives: Annotated[
        IncentiveScenario,
        typer.Argument(help="The incentive level to consider in the run."),
    ],
    year: Annotated[
        int,
        typer.Argument(
            callback=utils.year_callback,
            help="The assumption year for the analysis. Options are 2022, 2024, and 2025.",
        ),
    ],
    repository: Annotated[
        str, typer.Argument(help="Path to the dwind repository to use when running the model.")
    ],
    nodes: Annotated[
        int,
        typer.Argument(
            help="Number of HPC nodes or CPU nodes to run on. -1 indicates 75% of CPU limit."
        ),
    ],
    allocation: Annotated[str, typer.Argument(help="HPC allocation name.")],
    memory: Annotated[int, typer.Argument(help="Node memory, in GB (HPC only).")],
    walltime: Annotated[int, typer.Argument(help="Node walltime request, in hours.")],
    feature: Annotated[
        str,
        typer.Argument(
            help=(
                "Additional flags for the SLURM job, using formatting such as"
                " --qos=high or --depend=[state:job_id]."
            )
        ),
    ],
    env: Annotated[
        str,
        typer.Argument(
            help="The path to the dwind Python environment that should be used to run the model."
        ),
    ],
    model_config: Annotated[
        str, typer.Argument(help="Complete file name and path of the model configuration file")
    ],
    dir_out: Annotated[
        str, typer.Argument(help="Path to where the chunked outputs should be saved.")
    ],
    stdout_path: Annotated[
        Optional[str], typer.Option(help="The path to write stdout logs.")  # noqa
    ] = None,
    combine: Annotated[
        bool,
        typer.Option(
            "--combine/--no-combine",
            "-c/-C",
            help="Automatically combine the chunked results after analysis completion.",
        ),
    ] = True,
    remove_agent_chunks: Annotated[
        bool,
        typer.Option(
            "--remove-agent-chunks/--no-remove-agent-chunks",
            "-ra/-RA",
            help="Delete the temporary agent chunk files.",
        ),
    ] = True,
    remove_results_chunks: Annotated[
        bool,
        typer.Option(
            "--remove-results-chunks/--no-remove-results-chunks",
            "-rr/-RR",
            help="Delete the chunked results files. Ignored if --combine is not passed.",
        ),
    ] = True,
):
    """Run dwind via the HPC multiprocessing interface."""
    # sys.path.append(repository)
    from dwind.mp import MultiProcess

    # NOTE: collect_by_priority has been removed but may need to be reinstated
    mp = MultiProcess(
        location=location,
        sector=sector,
        scenario=scenario,
        incentive_scenario=incentives,
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

    agent_df = utils.load_agents(
        location=location, sector=sector, model_config=model_config, prepare=True
    )
    job_chunk_map = mp.run_jobs(agent_df)
    debug.job_summary(jobs=job_chunk_map.keys(), chunks=job_chunk_map.values())

    if remove_agent_chunks:
        utils.cleanup_chunks(dir_out, which="agents")

    if combine:
        if incentives:
            run_name = f"{location}_{sector}_{incentives}_{year}"
        else:
            run_name = f"{location}_{sector}_{scenario}_{year}"
        collect.combine_chunks(
            sector=Sector(sector),
            dir_out=dir_out,
            file_name=run_name,
            model_config=model_config,
            remove_results_chunks=remove_results_chunks,
        )


@app.command()
def interactive(
    location: Annotated[
        str, typer.Argument(help="The state, state_county, or priority region to run.")
    ],
    sector: Annotated[
        Sector, typer.Argument(help="One of fom (front of meter) or btm (back-of-the-meter).")
    ],
    scenario: Annotated[Scenario, typer.Argument(help="The scenario to run, such as 'baseline'.")],
    incentives: Annotated[
        IncentiveScenario,
        typer.Argument(help="The incentive level to consider in the run."),
    ],
    year: Annotated[
        int, typer.Argument(callback=utils.year_callback, help="The year basis of the scenario.")
    ],
    dir_out: Annotated[str, typer.Argument(help="save path")],
    model_config: Annotated[
        str, typer.Argument(help="Complete file name and path of the model configuration file")
    ],
):
    """Run dwind locally, or via an interactive session where a SLURM job is not scheduled."""
    sector = Sector(sector)
    agents = utils.load_agents(
        location=location, sector=sector.value, model_config=model_config, prepare=True
    )

    model = Model(
        agents=agents,
        location=location,
        sector=sector,
        scenario=scenario,
        incentive_scenario=incentives,
        year=year,
        out_path=dir_out,
        model_config=model_config,
    )
    model.run()


@app.command()
def config(
    config_path: Annotated[
        str, typer.Argument(help="Path to configuration TOML with run and model parameters.")
    ],
    use_hpc: Annotated[
        bool,
        typer.Option(help="Run via sbatch on the HPC (--use-hpc) or interactively (--no-use-hpc)."),
    ] = True,
    combine: Annotated[
        bool,
        typer.Option(
            "--combine/--no-combine",
            "-c/-C",
            help=(
                "Automatically combine the chunked results after analysis completion. Ignored"
                " if --no-use-hpc."
            ),
        ),
    ] = True,
    remove_agent_chunks: Annotated[
        bool,
        typer.Option(
            "--remove-agent-chunks/--no-remove-agent-chunks",
            "-ra/-RA",
            help="Delete the temporary agent chunk files. Ignored if --no-use-hpc.",
        ),
    ] = True,
    remove_results_chunks: Annotated[
        bool,
        typer.Option(
            "--remove-results-chunks/--no-remove-results-chunks",
            "-rr/-RR",
            help=(
                "Delete the chunked results files. Ignored if --combine is not passed or"
                " --no-use-hpc is passed."
            ),
        ),
    ] = True,
):
    """Run dwind via the HPC multiprocessing interface from a configuration file."""
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    print("Running the following configuration:")
    pprint(config)
    if use_hpc:
        hpc(
            combine=combine,
            remove_agent_chunks=remove_agent_chunks,
            remove_results_chunks=remove_results_chunks,
            **config,
        )
    else:
        interactive(**config)


@app.command()
def chunk(
    chunk_ix: Annotated[int, typer.Argument(help="Chunk number/index. Used for logging.")],
    location: Annotated[
        str, typer.Argument(help="The state, state_county, or priority region to run.")
    ],
    sector: Annotated[
        Sector, typer.Argument(help="One of fom (front of meter) or btm (back-of-the-meter).")
    ],
    scenario: Annotated[Scenario, typer.Argument(help="The scenario to run, such as baseline.")],
    incentives: Annotated[
        IncentiveScenario,
        typer.Argument(help="The incentive level to consider in the run."),
    ],
    year: Annotated[
        int, typer.Argument(callback=utils.year_callback, help="The year basis of the scenario.")
    ],
    out_path: Annotated[str, typer.Argument(help="save path")],
    repository: Annotated[
        str, typer.Argument(help="Path to the dwind repository to use when running the model.")
    ],
    model_config: Annotated[
        str, typer.Argument(help="Complete file name and path of the model configuration file")
    ],
):
    """Run a chunk of a dwind model. Internal only, do not run outside the context of a
    chunked analysis or debugging.
    """
    # Import the correct version of the library
    sys.path.append(repository)
    from dwind.model import Model

    agent_file = Path(out_path).resolve() / f"agent_chunks/agents_{chunk_ix}.pqt"
    agents = utils.load_agents(file_name=agent_file, model_config=model_config, prepare=False)

    model = Model(
        agents=agents,
        location=location,
        sector=sector,
        scenario=scenario,
        incentive_scenario=incentives,
        year=year,
        chunk_ix=chunk_ix,
        out_path=out_path,
        model_config=model_config,
    )
    model.run()


if __name__ == "__main__":
    app()
