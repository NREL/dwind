![dGen outputs in action](https://www.nrel.gov/analysis/dgen/assets/images/hero-hp-dgen.jpg)

## Watch The Webinar and Setup Tutorial
https://attendee.gotowebinar.com/recording/7790172234808601356
Note: the webinar is specific to the [Open Source dGen model](https://github.com/NREL/dgen), though much of the material applies for the dWind model.


## Get Your Tools
Install Docker
(Mac): https://docs.docker.com/docker-for-mac/install/; (Windows): https://docs.docker.com/docker-for-windows/install/

- Important: In Docker, go into Docker > Preferences > Resources and increase the allocation for disk size image for Docker. 16 GB is recommended for smaller (state-level) databases. 70+GB is required for restoring the national database. If you get a memory issue then you will need to increase the memory allocation and/or will need to prune past failed images/volumes. Running the Docker commands below will clear these out and let you start fresh:
```
   $ docker system prune -a
   $ docker volume prune -f
```
- Refer to Docker’s website for more details on this.

Install Anaconda Python 3.7 Version: https://www.anaconda.com/distribution/

Install PgAdmin: https://www.pgadmin.org/download/ (ignore all of the options for docker, python, os host, etc.)

Install Git: If you don't already have Git installed, then navigate here to install it for your operating system: https://www.atlassian.com/git/tutorials/install-git

Windows users: if you don't have UNIX commands enabled for command prompt/powershell then you'll need to install Cygwin or QEMU to run a UNIX terminal.

## Download Code 
New users should fork a copy of dGen to their own private github account 


Next, clone the repository to your local machine by running the following in a terminal/powershell/command prompt. Be sure to enter your username in place of `<github_username>` below, which will clone the forked repository:
```
   $ git clone https://github.com/<github_username>/dwind.git
```

- Create a new branch in this repository by running `git checkout -b <branch_name_here>`
- It is generally a good practice to leave the master branch of a forked repository unchanged for easier updating in future. Create new branches when developing features or performing configurations for unique runs.

# Running and Configuring dGen

### A. Create Environment
After cloning this repository and installing (and running) Docker as well as Anaconda, we will create our conda environment and Docker container:

1. Depending on directory you cloned this repo into, navigate in terminal to the python directory (/../dgen/python) and run the following command:

```
   $ conda env create -f dg3n.yml
```

- This will create the conda environment needed to run the dWind model. Note: this environment is identical to the [Open Source dGen](https://github.com/NREL/dgen) conda environment, for ease of use.

2. Create a container with PostgreSQL initialized:
```
   $ docker run --name postgis_1 -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -d mdillon/postgis
```
- Alternatively, if having issues connecting to the postgres server in pgAdmin, run:

```
   $ docker run --name postgis_1 -p 5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -d mdillon/postgis
```
- This will allow the docker container to select a different port to forward to 5432.

3. Connect to the PostgreSQL DB:

```
   $ docker container ls
   $ docker exec -it <container id> psql -U postgres
   $ postgres=# CREATE DATABASE dwind_db;
```
- If you get the error `psql: FATAL:  the database system is starting up` try rerunning the `docker exec` command again after a minute or so because Docker can take some time to initialize.

- `CREATE DATABASE` will be printed when the database is created. `\l` will display the databases in your server.

- `postgres=# \c dwind_db` can then be used to connect to the database, but this step is not necessary.


### B. Download data (agents and database):
Download data from the publicly available [Box folder](https://nrel.app.box.com/folder/123159108851). Make sure to unzip any zipped files once downloaded. We recommend starting with the database specific to the state you are interested in. We also recommended only downloading one data file at a time to avoid a "download size exceeded" error. 

Each state-level or national `.zip` file will contain the database backup as well as the files for both the residential and commercial agents.

Next, run the following in the command line (replacing `path_to_where_you_saved_database_file` below with the actual path where you saved your database file): 

```
   $ cat /path_to_where_you_saved_data/dwind_db.sql | docker exec -i <container id> psql -U postgres -d dwind_db
```

- Note, if on a Windows machine, use Powershell rather than command prompt. If Linux commands still aren't working in Powershell, you can copy the data to the docker container and then load the data by running:

```
   $ docker cp /path_to_where_you_saved_data/dwind_db.sql <container id>:/dwind_db.sql
   $ docker exec -i <container id> psql -U postgres -d dwind_db -f dwind_db.sql
```

- Backing up state-level databases will likely take 5-15 minutes. The national database will take 45-60 minutes. 
- Don't close docker at any point while running dWind.
- The container can be "paused" by running `$ docker stop <container id>` and "started" by running `$ docker start <container id>`

### C. Create Local Server:
Once the database is restored, open PgAdmin and create a new server. Name this whatever you want. Write "localhost" (or 127.0.0.1) in the host/address cell and "postgres" in both the username and password cells. Upon refreshing this and opening the database dropdown, you should be able to see your database. 

### D: Activate Environment 
Activate the `dg3n` environment and launch Spyder by opening a new terminal window and run the following command:

```
   $ conda activate dg3n
   $ (dg3n) spyder
```

- In Spyder, open the `dgen_model.py` file. This is what we will run once everything is configured.

### E: Configure Scenario
1. Open the blank input sheet located in `dwind_os/excel/input_sheet_os_dwind.xlsx`. This file defines most of the settings for a scenario. Configure it depending on the desired model run and save a copy in the input_scenarios folder, i.e. `dwind_os/input_scenarios/my_scenario.xlsx`. 

- See the Input Sheet Wiki page for more details on customizing scenarios.


2. In the python folder, open `pg_params_atlas.json` and configure it to your local database. If you did not change your username or password settings while setting up the Docker container, this file should look like the below example:

```
   {	
   	"dbname": "dwind_db",
    	"host": "localhost",
   	"port": "5432",
   	"user": "postgres",
   	"password": "postgres"
   }
```

- Localhost could also be set as "127.0.0.1"
- Save this file

The cloned repository will have already initialized the default values for the following important parameters:

* ` start_year = 2014 ` (in `dwind_os/python/config.py`)        --> year the model will begin at
* ` pg_procs = 2 ` (in `dwind_os/python/config.py`)             --> number of parallel processes the model will run with
* ` cores = 2 ` (in `dwind_os/python/config.py`)                --> number of cores the model will run with


### F: Run the Model
Run the model in the command line:
```
   $ python dgen_model.py
```
Or, open `dgen_model.py` in the Spyder IDE and hit the large green arrow "play button" near the upper left to run the model.

If `drop_output_schema = False` ( in `dwind_os/python/config.py`), results from the model run will be placed in a SQL table called "agent_outputs" within a newly created schema in the connected database. Because the database will not persist once a docker container is terminated, these results will need to be saved locally.

## Saving Results:
1. To backup the whole database, including the results from the completed run, please run the following command in terminal after changing the save path and database name:

```
   $ docker exec <container_id> pg_dumpall -U postgres > '/../path_to_save_directory/dwind_db.sql'
```

- This `.sql` file can be restored in the same way as was detailed above. 

2. To export just the "agent_outputs" table, simply right click on this table in PgAdmin and select the "Import/Export" option and configure how you want the data to be saved. Note, if a save directory isn't specified this will likely save in the home directory.