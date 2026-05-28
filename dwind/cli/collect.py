"""Enables running dwind as a CLI tool, the primary interface for working with dwind."""

from __future__ import annotations

from typing import Optional, Annotated
from pathlib import Path

import typer
import pandas as pd

from dwind.cli import utils
from dwind.config import Sector, Configuration


app = typer.Typer()


@app.command()
def combine_chunks(
    dir_out: Annotated[
        str,
        typer.Argument(
            help=(
                "Path to where the chunked outputs should be saved. Should be the same that was"
                " passed to the run command."
            )
        ),
    ],
    sector: Annotated[
        Sector, typer.Argument(help="One of fom (front of meter) or btm (back-of-the-meter).")
    ],
    model_config: Annotated[
        str, typer.Argument(help="Complete file name and path of the model configuration file")
    ],
    file_name: Annotated[
        Optional[str],  # noqa
        typer.Option(
            "--file-name",
            "-f",
            help="Custom filename, without the extension (e.g. .pqt), for the results data.",
        ),
    ] = None,
    remove_results_chunks: Annotated[
        bool,
        typer.Option(
            "--remove-results-chunks/--no-remove-results-chunks",
            "-rr/-RR",
            help="Delete the individual chunk files after saving the combined results.",
        ),
    ] = True,
):
    """Combine the results of a multi-job run based on the run's TOML configuration. Please note
    this has the potential to combine multiple runs as it does not respect the jobs ran during a
    single cycle.
    """
    dir_out = Path.cwd() if dir_out is None else Path(dir_out).resolve()
    out_path = dir_out / "chunk_files"
    result_files = [f for f in out_path.iterdir() if f.suffix == (".pqt")]
    model_config = Configuration(model_config)

    if len(result_files) == 0:
        print(f"No chunked results found in: {out_path}.")
        return None

    file_name = "results" if file_name is None else file_name
    result_agents = pd.concat([pd.read_parquet(f) for f in result_files])
    load_df = pd.read_csv(
        f"{model_config.load.DIR}/{model_config.load.LANDUSE}",
        usecols=["land_use", "application"],
    )
    result_agents.drop(columns="application").merge(load_df, on="land_use", how="left")
    apps = ["BTM, FOM", "BTM, FOM, Utility"]
    if sector is Sector.BTM:
        apps.append("BTM")
        drop_cols = ["index_x", "index_y"]
        # result_agents = result_agents.drop(columns=["consumption_hourly", "deprec_sch"])
    else:
        apps.extend(["FOM, Utility", "FOM"])
        drop_cols = ["lcoe_real_usd_kwh", "system_variable_om_per_kw", "cap_cost_multiplier"]

    result_agents = result_agents.loc[result_agents.application.isin(apps)]
    result_agents.loc[
        result_agents.state.eq("new_york"), ["state_abbr", "census_division_abbr"]
    ] = ["NY", "MA"]
    result_agents.loc[
        result_agents.state.eq("north_dakota"), ["state_abbr", "census_division_abbr"]
    ] = ["ND", "WNC"]
    result_agents.loc[
        result_agents.state.eq("south_dakota"), ["state_abbr", "census_division_abbr"]
    ] = ["SD", "WNC"]
    result_agents.loc[
        result_agents.state.eq("new_mexico"), ["state_abbr", "census_division_abbr"]
    ] = ["NM", "M"]
    result_agents.loc[
        result_agents.state.eq("new_hampshire"), ["state_abbr", "census_division_abbr"]
    ] = ["NH", "NE"]
    result_agents.loc[
        result_agents.state.eq("rhode_island"), ["state_abbr", "census_division_abbr"]
    ] = ["RI", "NE"]
    result_agents.loc[
        result_agents.state.eq("districtofcolumbia"), ["state_abbr", "census_division_abbr"]
    ] = ["DC", "SA"]
    result_agents.loc[
        result_agents.state.eq("west_virginia"), ["state_abbr", "census_division_abbr"]
    ] = ["WV", "SA"]
    result_agents.loc[
        result_agents.state.eq("north_carolina"), ["state_abbr", "census_division_abbr"]
    ] = ["NC", "SA"]
    result_agents.loc[
        result_agents.state.eq("south_carolina"), ["state_abbr", "census_division_abbr"]
    ] = ["SC", "SA"]

    result_agents = result_agents.drop(columns=drop_cols)
    f_out = dir_out / f"{file_name}.pqt"
    result_agents.to_parquet(f_out)
    print(f"Aggregated results saved to: {f_out}")
    if remove_results_chunks:
        for f in result_files:
            f.unlink()
            print(f"Removed: {f}")


@app.command()
def cleanup_agents(
    dir_out: Annotated[
        str,
        typer.Argument(
            help=(
                "Path to where the chunked agents were saved. Should be the same that was"
                " passed to the run command."
            )
        ),
    ],
):
    """Deletes the temporary agent chunk files generated at runtime.

    Args:
        dir_out (str): The base output directory, which should be the same as that passed to the
            run command.
    """
    utils.cleanup_chunks(dir_out, which="agents")


@app.command()
def cleanup_results(
    dir_out: Annotated[
        str,
        typer.Argument(
            help=(
                "Path to where the results agents were saved. Should be the same that was"
                " passed to the run command."
            )
        ),
    ],
):
    """Deletes the chunked results files generated at runtime.

    Args:
        dir_out (str): The base output directory, which should be the same as that passed to the
            run command.
    """
    utils.cleanup_chunks(dir_out, which="results")


if __name__ == "__main__":
    app()
