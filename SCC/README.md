# Running LearningSpikingDynamics on BU's SCC

The SCC uses Grid Engine (`qsub`/`qrsh`). The supplied job runs the current
`Main.py` on one CUDA GPU and saves the normal final output plus restartable
per-epoch files in `Epoch_Rasters/`.

## 1. Put the repository in project storage

From an SCC login node, use your actual project and username:

```bash
mkdir -p /projectnb/BU_PROJECT/USERNAME
cd /projectnb/BU_PROJECT/USERNAME
git clone REPOSITORY_URL LearningSpikingDynamics
cd LearningSpikingDynamics
```

If the repository is already present, update it with your normal Git workflow.
Before cloning, make sure the current local work has actually been committed and
pushed: the simulation currently depends on recently added files such as
`Simulation/initialize_from_mat.py` and `Pre_Processing/pre_cortical_handler_legacy_match.py`.
A fresh clone of an older branch will not contain those files. Alternatively,
transfer the complete working directory with SFTP/rsync rather than cloning.
Confirm that these data files were transferred:

```bash
ls Data/Targets/200k_target1.wav
ls Data/Data/all_units_info_with_polished_criteria_modified_perf.mat
ls Data/Data/silent_activity_matrix.npy
```

Do not run the simulation on a login node.

## 2. Test interactively once

Inspect currently shared GPUs:

```bash
qgpus -s -v
```

Request an interactive GPU and run the smoke test:

```bash
qrsh -P BU_PROJECT -l gpus=1 -l gpu_c=7.0 -l gpu_memory=24G -pe omp 4 -l mem_per_core=8G -l h_rt=01:00:00
module purge
module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt
cd /projectnb/BU_PROJECT/USERNAME/LearningSpikingDynamics
python SCC/check_scc_environment.py
exit
```

The smoke test checks CUDA, required packages and data paths, then constructs a
tiny version of the network. It does not run an optimization epoch.

## 3. Submit the full run

Submit from the repository root so `SGE_O_WORKDIR` points to the project:

```bash
cd /projectnb/BU_PROJECT/USERNAME/LearningSpikingDynamics
qsub -P BU_PROJECT SCC/run_gpu.qsub
```

The script requests one GPU with at least 24 GB VRAM and compute capability 7.0,
four CPU cores, 32 GB host RAM, and the SCC GPU maximum of 48 hours. For the
current one-cell `batch_size: 1200` run, this is a sensible starting request.
Use `qgpus -s -v` before tightening `gpu_type`, because a type constraint can
increase queue time.

Track or cancel the job with:

```bash
qstat -u "$USER"
qstat -j JOB_ID
qdel JOB_ID
```

The combined stdout/stderr log is written by Grid Engine in the submission
directory. During training, each `Epoch_Rasters/rasters_epoch_NNN.mat` is a full
epoch-boundary checkpoint.

## 4. Resume after interruption

Edit `simulation_config.yaml` on the SCC:

```yaml
parameter_initialization:
  from_mat:
    enabled: true
    path: "/projectnb/BU_PROJECT/USERNAME/LearningSpikingDynamics/Epoch_Rasters/rasters_epoch_020.mat"
    load_adam: true
```

Keep the same `batch_size`, `cell_targets`, `num_params`, learning rates and Adam
betas. `simulation.epochs` is the number of *additional* epochs. Submit the same
qsub script again. Output numbering continues from the restored Adam step.

## Notes

- The code uses one GPU; requesting multiple GPUs will not speed it up.
- `CUDA_VISIBLE_DEVICES` is assigned by SCC and must not be overwritten.
- `MPLBACKEND=Agg` prevents plotting code from requiring a graphical display.
- Checkpoints do not yet store NumPy/Torch RNG state. Parameters and Adam resume
  correctly, but a split stochastic run is not bit-for-bit identical.
- If `academic-ml/spring-2026` is retired, run `module avail academic-ml` and
  replace both module/environment names with the current PyTorch pair.
