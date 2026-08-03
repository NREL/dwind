"""Provides the :py:class:`ResourcePotential` class for gathering pre-calculated reV generation
data.
"""

from pathlib import Path

import h5py as h5
import pandas as pd

from dwind.config import Sector, Technology, Configuration


class ResourcePotential:
    """Helper class designed to retrieve pre-calculated energy generation data from reV."""

    def __init__(
        self,
        parcels: pd.DataFrame,
        model_config: Configuration,
        sector: Sector,
        tech: str = "wind",
        year: int = 2018,
    ):
        """Initializes the :py:class:`ResourcePotential` instance.

        Args:
            parcels (pd.DataFrame): The agent DataFrame containing at least the following columns:
                "gid", "rev_gid_{tech}", "solar_az_tilt" (solar only), "azimuth_{sector}"
                (solar only), "tilt_{tech}" (solar only), "turbine_class" (wind only),
                "wind_turbine_kw" (wind only), and "turbine_height_m" (wind only).
            model_config (Configuration): The pre-loaded model configuration data object containing
                the requisite SQL, file, and configuration data.
            sector (dwind.config.Sector): A valid sector instance.
            tech (str, optional): One of "solar" or "wind". Defaults to "wind".
            year (int, optional): Resource year for the reV lookup. Defaults to 2018.

        Raises:
            ValueError: Raised if :py:attr:`parcels:` is missing any of the required columns.
        """
        self.df = parcels
        self.tech = Technology(tech)
        self.sector = sector
        self.year = year
        self.config = model_config

        solar_cols = ("solar_az_tilt", f"azimuth_{self.sector.value}", f"tilt_{self.tech.value}")
        # wind_cols = ("turbine_class", "wind_turbine_kw", "turbine_height_m")
        wind_cols = ("wind_turbine_kw", "turbine_height_m")

        if self.tech is Technology.WIND:
            cols = wind_cols
        elif self.tech is Technology.SOLAR:
            cols = solar_cols

        missing = set(cols).difference(self.df.columns.tolist())
        if missing:
            raise ValueError(f"`parcels` is missing the following columns: {', '.join(missing)}")

    def create_rev_gid_to_summary_lkup(
        self, configs: list[str], *, save_csv: bool = True
    ) -> pd.DataFrame:
        """Creates the reV summary tables based on the "gid" mappings in :py:attr:`parcels`.

        Args:
            configs (list[str]): The list of technology-specific configurations where the generation
                data should be retrieved.
            save_csv (bool, optional): If True, save the resulting lookup calculated from reV to the
                reV folder definied in ``Configuration.rev.generation.{tech}_DIR``. Defaults to
                True.

        Returns:
            pd.DataFrame: reV generation lookup table for the technology-specific configurations in
                :py:attr:`configs`.
        """
        config_dfs = []
        for c in configs:
            file_str = self.config.rev.DIR / f"rev_{c}_generation_{self.year}.h5"

            with h5.File(file_str, "r") as hf:
                rev_index = pd.DataFrame(hf["meta"][...]).index.to_series()
                gids = pd.DataFrame(hf["meta"][...])[["gid"]]
                annual_energy = pd.DataFrame(hf["annual_energy"][...])
                cf_mean = pd.DataFrame(hf["cf_mean"][...])

            config_df = pd.concat([rev_index, gids, annual_energy, cf_mean], axis=1)
            config_df.columns = [
                f"rev_index_{self.tech.value}",
                f"rev_gid_{self.tech.value}",
                f"{self.tech.value}_naep",
                f"{self.tech.value}_cf",
            ]

            config_df["config"] = c
            config_dfs.append(config_df)

        summary_df = pd.concat(config_dfs)

        if save_csv:
            save_name = (
                self.config.rev.generation[f"{self.tech.value}_DIR"]
                / f"lkup_rev_gid_to_summary_{self.tech.value}_{self.year}.csv"
            )
            summary_df.to_csv(save_name, index=False)

        return summary_df

    def find_rev_summary_table(self):
        """Creates the generation summary data for each of the :py:attr:`tech`-specific
        configurations specified in :py:attr:`config.rev.settings.{tech}`, then maps it to the
        agent data (:py:attr:`parcels`), overwriting any previously computed data.
        """
        if self.tech is Technology.SOLAR:
            configs = self.config.rev.settings.solar
            config_col = "solar_az_tilt"
            col_list = ["gid", f"rev_gid_{self.tech.value}", config_col]
            self.df[config_col] = self.df[f"azimuth_{self.sector.value}"].map(
                self.config.rev.settings.azimuth_direction_to_degree
            )
            self.df[config_col] = (
                self.df[config_col].astype(str)
                + "_"
                + self.df[f"tilt_{self.tech.value}"].astype(str)
            )
        elif self.tech is Technology.WIND:
            configs = self.config.rev.settings.wind
            config_col = "turbine_class"
            col_list = [
                "gid",
                f"rev_gid_{self.tech.value}",
                config_col,
                "turbine_height_m",
                "wind_turbine_kw",
            ]
            self.df[config_col] = self.df["wind_turbine_kw"].map(self.config.rev.turbine_class_dict)

        out_cols = [
            *col_list,
            f"rev_index_{self.tech.value}",
            f"{self.tech.value}_naep",
            f"{self.tech.value}_cf",
        ]

        drop_cols = [
            f"rev_gid_{self.tech.value}",
            f"{self.tech.value}_naep",
            f"{self.tech.value}_cf",
        ]
        self.df = self.df.drop(columns=[c for c in drop_cols if c in self.df])
        k = f"{self.tech.value}_DIR"
        f_gen = (
            f"{self.config.rev.generation[k]}/"
            f"lkup_rev_gid_to_summary_{self.tech.value}_{self.year}.csv"
        )

        if ("s3://" in f_gen) or Path(f_gen).exists():
            generation_summary = pd.read_csv(f_gen, dtype_backend="pyarrow")
        else:
            generation_summary = self.create_rev_gid_to_summary_lkup(configs)

        generation_summary = (
            generation_summary.reset_index(drop=True)
            .drop_duplicates(subset=[f"rev_index_{self.tech.value}", "config"])
            .rename(columns={"config": config_col})
        )
        agents = self.df.merge(
            generation_summary, how="left", on=[f"rev_index_{self.tech.value}", config_col]
        )
        return agents[out_cols]

    def prepare_agents_for_gen(self):
        """Create lookup column based on each technology."""
        if self.tech is Technology.WIND:
            # drop wind turbine size duplicates
            # SINCE WE ASSUME ANY TURBINE IN A GIVEN CLASS HAS THE SAME POWER CURVE
            self.df.drop_duplicates(subset=["gid", "wind_size_kw"], keep="last", inplace=True)
            # if running FOM sector, only consider a single (largest) turbine size
            if self.sector is Sector.FOM:
                self.df = self.df.loc[self.df["wind_size_kw"] == self.df["wind_size_kw_fom"]]

            self.df["turbine_class"] = self.df["wind_turbine_kw"].map(
                self.config.rev.turbine_class_dict
            )

        if self.tech is Technology.SOLAR:
            # NOTE: tilt and azimuth are sector-specific
            self.df["solar_az_tilt"] = self.df[f"azimuth_{self.sector.value}"].map(
                self.config.rev.settings.azimuth_direction_to_degree
            )
            self.df["solar_az_tilt"] = self.df["solar_az_tilt"].astype(str)
            self.df["solar_az_tilt"] = (
                self.df["solar_az_tilt"] + "_" + self.df[f"tilt_{self.sector.value}"].astype(str)
            )

    def merge_gen_to_agents(self, tech_agents: pd.DataFrame):
        """Merges :py:attr:`tech_agents` to the parcel data :py:attr:`df`.

        Args:
            tech_agents (pd.DataFrame): The technology-specific energy generation data.
        """
        if self.tech is Technology.WIND:
            cols = ["turbine_height_m", "wind_turbine_kw", "turbine_class"]
        else:
            # NOTE: need to drop duplicates in solar agents
            # since multiple rows exist due to multiple turbine configs for a given parcel
            tech_agents = tech_agents.drop_duplicates(
                subset=["gid", "rev_gid_solar", "solar_az_tilt"]
            )
            cols = ["solar_az_tilt"]

        cols.extend(["gid", f"rev_index_{self.tech.value}"])

        self.df = self.df.merge(tech_agents, how="left", on=cols)

    def match_rev_summary_to_agents(self):
        """Runs the energy generation gathering and merging steps, and retursns back the updated
        :py:attr:`df` agent/parcel data.

        Returns:
            pd.DataFrame: Updated agent/parcel data with rec/alculated "wind_aep" or "solar_aep"
                information for each agent.
        """
        self.prepare_agents_for_gen()
        tech_agents = self.find_rev_summary_table()
        self.merge_gen_to_agents(tech_agents)

        if self.tech is Technology.WIND:
            # fill nan generation values
            self.df = self.df.loc[
                ~((self.df["wind_naep"].isnull()) & (self.df["turbine_class"] != "none"))
            ]
            self.df["wind_naep"] = self.df["wind_naep"].fillna(0.0)
            self.df["wind_cf"] = self.df["wind_cf"].fillna(0.0)
            # self.df['wind_cf_hourly'] = self.df['wind_cf_hourly'].fillna(0.)
            # calculate annual energy production (aep)
            self.df["wind_aep"] = self.df["wind_naep"] * self.df["wind_turbine_kw"]
            # self.df = self.df.drop(columns="turbine_class")
        elif self.tech is Technology.SOLAR:
            # fill nan generation values
            self.df = self.df.loc[~(self.df["solar_naep"].isnull())]
            # size groundmount system to equal wind aep
            # self.df['solar_size_kw_fom'] = np.where(
            # self.df['solar_groundmount'],
            # self.df['wind_aep'] / (self.df['solar_cf'] * 8760),
            # self.df['solar_size_kw_fom']
            # )

            # calculate annual energy production (aep)
            self.df["solar_aep"] = self.df["solar_naep"] * self.df["solar_size_kw_fom"]

        return self.df
