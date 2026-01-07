# Monte Carlo Simulation

This directory contains the code for the Monte Carlo simulation study reported in the paper.  
The study compares the proposed MMH-based approach against a nominal model-based approach and additional baseline methods across multiple randomized runs.

## Monte Carlo Study on HPC (Docker + Apptainer + Slurm)

For the full Monte Carlo study consisting of 100 independent runs, we recommend using the provided containerized workflow. This setup ensures reproducibility and reliable execution on HPC clusters.

### 1. Build the Docker image locally

From the **repository root**, build the Docker image:

```bash
docker build \
  -f .devcontainer/Dockerfile \
  -t mc_ode_mmh_container:latest \
  .
```

#### Note on linear solvers:
The container is configured to use the proprietary HSL linear solvers for improved numerical performance.
To include them, place `HSL_jll.jl.v2024.11.28.tar.gz` into the `.devcontainer/` directory next to the Dockerfile before building the image.

If HSL is not available, remove the corresponding lines from the Dockerfile. The code will still run using open-source solvers (e.g., MUMPS), but execution may be slower and numerical results may differ slightly.

### 2. Save and transfer the Docker image to the cluster

Create a tar archive of the image:

```bash
docker save mc_ode_mmh_container:latest -o mc_ode_mmh_container.tar
```

Transfer it to the cluster:

```bash
scp mc_ode_mmh_container.tar <cluster_user>@<cluster_address>:/path/to/project/root
```

Replace `/path/to/project/root` with the desired project directory on the cluster.

### 3. Build the Apptainer image on the cluster

On the cluster, load Apptainer and build the `.sif` image:

```bash
. /etc/profile.d/modules.sh
module load apptainer

apptainer build /path/to/project/root/mc_ode_mmh_container.sif \
    docker-archive:///path/to/project/root/mc_ode_mmh_container.tar
```

Use the same project path as in the previous step.

### 4. Configure paths in the Slurm script (important)

The Slurm script uses bind mounts to map host directories into the Apptainer container:

- Host project directory → `/home/developer/workspace`
- Host .julia depot → `/home/developer/.julia`

Since the container filesystem is read-only, Julia must use a writable depot directory on the host.
Each user must therefore adapt the bind paths manually.

Open `experiments/montecarlo/run_montecarlo_slurm.sh` and replace the example bindings

```
--bind /sq/home/ge82sem/ifac2026:/home/developer/workspace \
--bind /sq/home/ge82sem/ifac2026/.julia:/home/developer/.julia \
```

with paths appropriate for your system, for example:

```
--bind /your/local/path/to/repository/root:${PROJECT_DIR} \
--bind /your/local/path/to/repository/root/.julia:/home/developer/.julia \
```

Create the required host directories (only needed once):

```bash
mkdir -p /your/local/path/to/repository/root/.julia
mkdir -p /your/local/path/to/repository/root/experiments/montecarlo/logs
mkdir -p /your/local/path/to/repository/root/experiments/montecarlo/results
```

These directories store:
- Julia precompile files and package caches,
- Slurm log output,
- Monte Carlo simulation results.

### 5. Instantiate the Julia environment

Before submitting the Slurm job for the first time, instantiate the Julia project inside the container so that all dependencies are installed into the bound Julia depot:

```bash
apptainer exec \
    --cleanenv \
    --bind /your/local/path/to/repository/root:/home/developer/workspace \
    --bind /your/local/path/to/repository/root/.julia:/home/developer/.julia \
    /your/local/path/to/repository/root/mc_ode_mmh_container.sif \
    julia -e 'using Pkg;
              Pkg.activate();
              Pkg.develop(path=ENV["HSL_PATH"], shared=true);
              Pkg.activate("/home/developer/workspace");
              Pkg.instantiate()'
```

This step only needs to be performed once per environment.

### 6. Run the Monte Carlo simulation via Slurm

From the repository root on the cluster, submit the Slurm job:

```bash
sbatch experiments/montecarlo/run_montecarlo_slurm.sh
```

This submits the job array defined in the script and executes the Monte Carlo simulations.
Output is written to:

- `experiments/montecarlo/logs/`
- `experiments/montecarlo/results/`

## Analyzing Results

Once all simulations have completed, the results can be aggregated and analyzed using the provided analysis script.

### Configure the analysis script

Open `experiments/montecarlo/analyze_montecarlo.jl` and set the results directory, for example:

```julia
const RESULTS_DIR = joinpath(@__DIR__, "results", "test_run")
```

### Run the analysis

```bash
julia --project=. experiments/montecarlo/analyze_montecarlo.jl
```

This script:

- aggregates performance metrics (costs, constraint violations) across all runs,
- prints a summary table to the console,
- generates plots for selected seeds (configured via `seeds_to_plot` in the script).