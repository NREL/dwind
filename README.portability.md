## Portability

This readme describes how to run this code on different systems:

### UCAR derecho

```
# on derecho
module load conda
git clone git clone https://github.com/NatLabRockies/dwind.git
cd dwind
conda activate dwind
pip install -e .[dev]

mkdir -p /glade/work/calebp/dwind_data/agents
mkdir -p /glade/work/calebp/dwind_data/cambium_processed
mkdir -p /glade/work/calebp/dwind_configs/rev/wind
mkdir -p /glade/work/calebp/dwind_configs/costs/atb24
mkdir -p /glade/work/calebp/dwind_configs/sizing/wind
mkdir -p /glade/work/calebp/dwind_data/tariffs
mkdir -p /glade/work/calebp/dwind_data/consumption/2024
mkdir -p /glade/work/calebp/dwind_data/nem
mkdir -p /glade/work/calebp/dwind_data/incentives

# on kestrel
rsync -avc /projects/dwind/agents/colorado_larimer derecho:/glade/work/calebp/dwind_data/agents/
scp /projects/dwind/data/cambium_processed/Cambium24_MidCase_pca_2025_processed.pqt derecho:/glade/work/calebp/dwind_data/cambium_processed/
scp /projects/dwind/data/rev/rev_*_generation_2018.h5 /glade/work/calebp/dwind_configs/rev/wind/
scp /projects/dwind/configs/rev/wind/lkup_rev_index_2012_to_2018.csv derecho:/glade/work/calebp/dwind_configs/rev/wind/
scp /projects/dwind/configs/rev/wind/lkup_rev_gid_to_summary_*.csv derecho:/glade/work/calebp/dwind_configs/rev/wind/
rsync -avc /projects/dwind/configs/costs/atb24/ derecho:/glade/work/calebp/dwind_configs/costs/atb24
scp /projects/dwind/data/tariffs/2025_tariffs.pqt derecho:/glade/work/calebp/dwind_data/tariffs/
scp /projects/dwind/data/crb_consumption_hourly.h5 derecho:/glade/work/calebp/dwind_data/
scp /projects/dwind/data/parcel_landuse_load_application_mapping.csv derecho:/glade/work/calebp/dwind_data/
scp /projects/dwind/data/consumption/2024/load_scaling_factors.csv derecho:/glade/work/calebp/dwind_data/consumption/2024/
scp /projects/dwind/data/county_nerc_join.csv derecho:/glade/work/calebp/dwind_data/
scp /projects/dwind/data/consumption/aeo_load_growth_projections_nerc_2023_updt.csv derecho:/glade/work/calebp/dwind_data/consumption/
scp /projects/dwind/configs/sizing/wind/lkup_block_to_pgid_2020.csv derecho:/glade/work/calebp/dwind_configs/sizing/wind/
scp /projects/dwind/data/nem/nem_baseline_2025.csv derecho:/glade/work/calebp/dwind_data/nem/
scp /projects/dwind/data/incentives/2025_incentives.pqt derecho:/glade/work/calebp/dwind_data/incentives/

# on derecho
qinteractive -A UCSG0003  -l walltime=01:00:00
module load conda
cd dwind
conda activate dwind
dwind run config examples/larimer_county_btm_baseline_2025.toml --no-use-hpc
```

### AWS Setup (on derecho)

```
module load conda
conda activate dwind
pip install awscli s3fs

# fill ~/.aws/credentials

# initial data upload
aws s3 sync /glade/work/calebp/dwind_configs s3://dwind-data-test/
aws s3 sync /glade/work/calebp/dwind_data s3://dwind-data-test/
```
