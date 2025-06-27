import h5py as h5
import pandas as pd

from dwind import Configuration


class ResourcePotential:
    def __init__(
        self, parcels, model_config: Configuration, tech="wind", application="fom", year="2018"
    ):
        self.df = parcels
        self.tech = tech
        self.application = application
        self.year = year
        self.config = model_config

        if self.tech not in ("wind", "solar"):
            raise ValueError("`tech` must be one of 'solar' or 'wind'.")

    def create_rev_gid_to_summary_lkup(self, configs, save_csv=True):
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
                f"rev_index_{self.tech}",
                f"rev_gid_{self.tech}",
                f"{self.tech}_naep",
                f"{self.tech}_cf",
            ]

            config_df["config"] = c
            config_dfs.append(config_df)

        summary_df = pd.concat(config_dfs)

        if save_csv:
            save_name = (
                self.config.rev.generation[f"{self.tech}_DIR"]
                / f"lkup_rev_gid_to_summary_{self.tech}_{self.year}.csv"
            )
            summary_df.to_csv(save_name, index=False)

        return summary_df

    def find_rev_summary_table(self):
        if self.tech == "solar":
            configs = self.config.rev.settings.solar
            config_col = "solar_az_tilt"
            col_list = ["gid", f"rev_gid_{self.tech}", config_col]
            self.df[config_col] = self.df[f"azimuth_{self.application}"].map(
                self.config.rev.settings.azimuth_direction_to_degree
            )
            self.df[config_col] = (
                self.df[config_col].astype(str) + "_" + self.df[f"tilt_{self.tech}"].astype(str)
            )
        elif self.tech == "wind":
            configs = self.rev.settings.wind
            config_col = "turbine_class"
            col_list = [
                "gid",
                f"rev_gid_{self.tech}",
                config_col,
                "turbine_height_m",
                "wind_turbine_kw",
            ]
            self.df[config_col] = self.df["wind_turbine_kw"].map(self.config.rev.turbine_class_dict)

        out_cols = [*col_list, f"rev_index_{self.tech}", f"{self.tech}_naep", f"{self.tech}_cf"]

        f_gen = (
            self.config.rev.generation[f"{self.tech}_DIR"]
            / f"lkup_rev_gid_to_summary_{self.tech}_{self.year}.csv"
        )

        if f_gen.exists():
            generation_summary = pd.read_csv(f_gen)
        else:
            generation_summary = self.create_rev_gid_to_summary_lkup(configs)

        generation_summary = (
            generation_summary.reset_index(drop=True)
            .drop_duplicates(subset=[f"rev_index_{self.tech}", "config"])
            .rename(columns={"config": config_col})
        )
        agents = self.df.merge(
            generation_summary, how="left", on=[f"rev_index_{self.tech}", config_col]
        )
        return agents[out_cols]

    def prepare_agents_for_gen(self):
        # create lookup column based on each tech
        if self.tech == "wind":
            # drop wind turbine size duplicates
            # SINCE WE ASSUME ANY TURBINE IN A GIVEN CLASS HAS THE SAME POWER CURVE
            self.df.drop_duplicates(subset=["gid", "wind_size_kw"], keep="last", inplace=True)
            # if running FOM application, only consider a single (largest) turbine size
            if self.application == "fom":
                self.df = self.df.loc[self.df["wind_size_kw"] == self.df["wind_size_kw_fom"]]

            self.df["turbine_class"] = self.df["wind_turbine_kw"].map(
                self.config.rev.turbine_class_dict
            )

        if self.tech == "solar":
            # NOTE: tilt and azimuth are application-specific
            self.df["solar_az_tilt"] = self.df[f"azimuth_{self.application}"].map(
                self.config.rev.settings.azimuth_direction_to_degree
            )
            self.df["solar_az_tilt"] = self.df["solar_az_tilt"].astype(str)
            self.df["solar_az_tilt"] = (
                self.df["solar_az_tilt"] + "_" + self.df[f"tilt_{self.application}"].astype(str)
            )

    def merge_gen_to_agents(self, tech_agents):
        if self.tech == "wind":
            cols = ["turbine_height_m", "wind_turbine_kw", "turbine_class"]
        else:
            # NOTE: need to drop duplicates in solar agents
            # since multiple rows exist due to multiple turbine configs for a given parcel
            tech_agents = tech_agents.drop_duplicates(
                subset=["gid", "rev_gid_solar", "solar_az_tilt"]
            )
            cols = ["solar_az_tilt"]

        cols.extend(["gid", f"rev_index_{self.tech}"])

        self.df = self.df.merge(tech_agents, how="left", on=cols)

    def match_rev_summary_to_agents(self):
        self.prepare_agents_for_gen()
        tech_agents = self.find_rev_summary_table()
        self.merge_gen_to_agents(tech_agents)

        if self.tech == "wind":
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
        else:
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
