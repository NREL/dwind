# Changelog

## v0.5 - 3 June 2026

- Fix bugs from hackathon (file paths, breakeven calculations)

## v0.4.1 - 28 April 2026

- Fixes a typo in the new GitHub repository's name.

## v0.4 - 28 April 2026

- Code updated to be compatible with PySAM 7.1+ releases.
  - Updates the `pyproject.tom` and `environment.yml` specifications so internal users
    can run the model without additional setup steps.
  - Default configurations updated.
  - Utilizes `from_existing` function to share data with existing PySAM classes.
  - Minor bug fixes and logging cleanup.
- Adds an `IncentiveScenario` to all run configuration options for "standard" and "no incentives".
- Documentation updates:
  - Adds a contributor's guide.
  - Adds a PR template to better document changes made through PRs.
  - Adds a CI check to ensure the docs build before changes can be made to the main or dev branchces.

## v0.3.3 - 12 September 2025

- Creates the documentation site for an API reference guide on both the CLI and underlying code.
- Removes extraneous files carried over from internal analysis work.
- Updates the dependency stack to support building the documentation.
- Fixes miscellaneous typing and docstring issues.

## v0.3.2 - 11 September 2025

- Adds docstrings for the core model for every module, class, method, and function.
- Adds `Enum` classes for all standard validation data to simplify reused string-based `if`/`elif`/`else` blocks.
- Moves the `Enum` validation classes for the CLI to `dwind/config.py`.
- Fixes an incorrect data assumption where 2012 replaced the standard 2018 reV year basis.
- Fixes issues with Typer CLI interface caused by `Enum` misconfigurations, and consistently applies
  the `Enum`-based input validation across all run commands.
- Significant improvements to the memory efficiency of the core code through the following changes:
  - Converting the CRB energy consumption data to HDF data using an integer-based multi-index that
    is allows easy conversion to the text-based descriptors using the `Enum` `CRBModel`. Using
    HDF data allows for single line access to the annual, hourly consumption data (8760 x 1) based
    each agent's computed specifiers, rather than storing the consumption data as a list in a
    dataframe column.
  - Converts more model data to Parquet file for faster I/O.
  - Reduces extra allocations in results compilations of `ProcessPoolExecutor`.
  - `dwind/valuation.py::ValueFunctions.find_cf_from_rev_wind` converts the data to lists instead
    of mapping to nested tuples.
  - Create base list data using `[(val1, val2)] * N` instead of `[(val1, val2) for i in range(len(base_data))]` to reduce redundant computation and excessive allocations.
  - Continue vectorizing pandas operations.
  - Converts most of `dwind/utils/array.py`'s custom scaling and interpolation functions to
    vectorized applications where the code is being called.
  - Converts the Pandas `dtype_backend` to PyArrow in place of NumPy, reducing the size of each
    dataframe in memory to 30% of its original size.
- Updated, and more advanced CLI with commands `run` `debug` and `collect` that allow for atomized
  functionality. Add the `--help` flag to any of these commands to show the available options.
  - `run`: run single or multiple job analyses (`run`), or (re)run individual chunks/jobs. Multiple
    job runs will automatically display the final node status and associated chunk. Options also
    exist to cleanup chunked data and combine results at runtime.
  - `debug`: Get completed job status(es) and identify missing agents in the results.
  - `collect`: Combine results data, and delete chunked agent and results files.
- Added the `dwind/cli` subpackage to keep the CLI-specific code in one place, and enable a cleaner
  package with single module for each available command.
- Common functionality is moved out of `dwind/cli/run.py` and placed in `dwind/cli/utils.py` for
  cleaner reuse.

## v0.3.1 - 23 July 2025

- Fixes various small bugs introduced in v0.3, primarily typos, or incorrect references
  that were unable to be properly debugged while Kestrel was down.
- Enable Python 3.9 compatibility while PySAM upgrade is ongoing, and the dwind-compatible version
  cannot be installed.
- Temporarily caps the maximum number of nodes to 80% of the maximum or requested amount to help
  with memory management in the FOM cases.
- Updates the job status printing to a user-friendly rich live table that is updated every 5 seconds
  to show the waiting time for a job to run, and the actual run time.
- Reduces the memory usage across the FOM case throughout the `dwind/valuation.py` methods, with
  the following major contributors:
  - Replaces data frames mapping from arrays to tuples to a direct `DataFrame.values.tolist()`.
  - Extracts PySAM outputs, then filters out the variables that were not requested, rather than
    calling the output getter for each requested variable.
  - Merges the census tract data at analysis preparation time, rather than analysis run time.
- Improves code readability for pandas assignments.
- Creates `dwind/utils/` for array methods (`array.py`) and hpc job status logging methods
  (`hpc.py`).
- Adds a missing job status code.

## v0.3 - 27 June 2025

This is the fist step of a major refactor, and introduces significant breaking changes. As such,
please update your cloned installation using `pip install -e /path/to/dwind` and `dwind --install-completion`.

- New CLI interface for running the model, powered by Typer. For more information on available
  commands, run `dwind --help`, but the primary options are as listed below.
  - `dwind run`: run an analysis in an interactive session, or without starting a SLURM-scheduled `sbatch` job.
  - `dwind run-config`: run an analysis as an`sbatch`job via SLURM using a TOML configuration file for parameterization.
  - `dwind run-hpc`: same as above, except with passing all arguments to the command line.
  - `dwind run-chunk`: run a single chunk of data as an `sbatch` job via SLURM. This is primarily an interface for `run-config` and `run-hpc` to kick off many smaller jobs.
- Removes a significant amount of outdated and unused code from the repository, and restructures the
  project to take on the form of a standard Python package.
- Improves many of the docstrings to provide relevant information about inputs and outputs.
- Replaces the usage pickled dataframes with Parquet files to reduce memory consumption and ensure
  data can be loaded consistently regardless of NumPy, Pandas, or Python versions.
- Significant peedups for both the BTM and FOM analysis flows by adopting more efficient Pandas
  usage, vectorization, data handling, and refined I/O processes.
- Upgrades to modern Pandas (2.x) and NumPy (2.x).
- Replaces the use `dwind/python/config.py` with a configuration TOML file that can contains core
  model configurations, data storage locations, and SQL credentials. This TOML file should be stored
  with the analysis data. To support the varying configurations of the data, there is a custom `dwind/config.py::Configuration` class that auto populates the user and password details in the [sql] section with the provided information in the same [sql] block. Additionally, the rev dictionary values for turbine size mappings are also automatically mapped to float data, and any field with "DIR" in it will be converted into a `pathlib.Path` object.
