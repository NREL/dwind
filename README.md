# Distributed Wind Generation Model (dWind)

Please note that at this time the model can only be run on NREL's Kestrel HPC system. Though a
savvy user could recreate our data in their own computing environment and update the
internal pointers in the example configuration at `examples/larimer_county_btm_baseline_2025.toml`
and `examples/model_config.toml`.

## Installing dwind

1. Install Anaconda or Miniconda (recommended) if not already installed.
2. Clone the repository

   ```bash
   git clone https://github.com/NREL/dwind.git
   ```

3. Navigate to the dwind repository.

   ```bash
   cd /path/to/dwind
   ```

4. Create your dwind environment using our recommended settings and all required dependencies.

    ```bash
    conda env create -f environment.yml
    ```

## Running

### Configuring

`dWind` relies on 2 configuration files: 1) a system-wise setting that can be shared among a team,
and 2) a run-specific configuration file. Both will be described below.

#### Primary model configuration

The primary model configuration should look exactly like (or be compatible with)
`examples/model_config.toml` to ensure varying fields are read correctly throughout the model.

Internally, `dWind` is able to convert the following data to adhere to internal usage:
- Any field with "DIR" is converted to a Python `pathlib.Path` object for robust file handling
- SQL credentials and constructor strings are automatically formed in the `[sql]` table for easier
  construction of generic connection strings. Specifically the `{USER}` and `{PASSWORD}` fields
  get replaced with their corresponding setting in the same table.

`Configuration`, the primary class handling this data allows for dot notation and dictionary-style
attribute calling at all levels of nesting. This means, `config.pysam.outputs.btm` and
`config.pysam.outputs["btm"]` are equivalent. This makes for more intuitive dynamic attribute
fetching when updating the code for varying cases.

#### Run configuration

The run-specific configuration should look like `examples/larimer_county_btm_baseline_2025.toml`,
which controls all the dynamic model settings, HPC configurations, and a pointer to the primary
model configuration described above.

### Run the model

`dwind` has a robust CLI interface allowing for the usage of `python path/to/dwind/dwind/run.py` or
`dwind`. The model currently supports the run prompts:
- `dwind run-config <run-configuration.toml>
- `dwind run-hpc --arg1 ... --argn` where the `--arg` parameters are used in place of the run TOML
  file.

If at any point, further guidance is needed, pass `--help` to `dwind` or any of the subcommands for
detailed information on the required and optional inputs.

To run the model, it is recommended to use the following workflow from your analysis folder.
1. Start a new `screen` session on Kestrel.

   ```bash
   screen -S <analysis-name>
   ```

2. Load your conda environment with dwind installed.

   ```bash
   module load conda
   conda activate <env_name>
   ```

3. Navigate to your analysis folder if your relative data locations in your run configuration are
   relative to the analysis folder.

   ```bash
   cd /path/to/analysis/location
   ```

4. Run the model.

   ```bash
   dwind run-config examples/larimer_county_btm_baseline_2025.toml
   ```

5. Disconnect your screen `Ctrl` + `a` + `d` and wait for the analysis to complete and view your
   results.

## `dwind` run settings

### `run-config`

```bash
*    config_path      TEXT  Path to configuration TOML with run and model parameters. [default: None] [required]
```

### `run-hpc`

```bash
 *  --location            TEXT        The state, state_county, or priority region to run. [default: None] [required]
 *  --sector              [fom|btm]   One of fom (front of meter) or btm (back-of-the-meter). [default: None] [required]
 *  --scenario            [baseline]  The scenario to run (baseline is the current only option). [default: None] [required]
 *  --year                INTEGER     The assumption year for the analysis. Options are 2022, 2024, and 2025. [default: None] [required]
 *  --repository          TEXT        Path to the dwind repository to use when running the model. [default: None] [required]
 *  --nodes               INTEGER     Number of HPC nodes or CPU nodes to run on. -1 indicates 75% of CPU limit. [default: None] [required]
 *  --allocation          TEXT        HPC allocation name. [default: None] [required]
 *  --memory              INTEGER     Node memory, in GB (HPC only). [default: None] [required]
 *  --walltime            INTEGER     Node walltime request, in hours. [default: None] [required]
 *  --feature             TEXT        Additional flags for the SLURM job, using formatting such as --qos=high or --depend=[state:job_id]. [default: None] [required]
 *  --env                 TEXT        The path to the dwind Python environment that should be used to run the model. [default: None] [required]
 *  --model-config        TEXT        Complete file name and path of the model configuration file [default: None] [required]
 *  --dir-out             TEXT        Path to where the chunked outputs should be saved. [default: None] [required]
 *  --stdout-path         TEXT        The path to write stdout logs. [default: None]
```
