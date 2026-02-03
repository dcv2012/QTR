from pathlib import Path
from src.models.GMM.GMM_experiment import run_gmm_experiments
from src.models.PMMR.PMMR_experiment import run_pmmr_experiments
from src.models.MinMax.MinMax_experiment import run_minmax_experiments
from src.models.MMR.MMR_experiment import run_mmr_experiments, run_mmr_rhc

# Simulation

# GMM
# run_gmm_experiments(100, 'S1', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S2', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S3', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S4', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S5', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S6', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))


# run_gmm_experiments(100, 'S1', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S2', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S3', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S4', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S5', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
# run_gmm_experiments(100, 'S6', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))


# PMMR
# run_pmmr_experiments(100, 'S1', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S2', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S3', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S4', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S5', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S6', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))


# run_pmmr_experiments(100, 'S1', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S2', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S3', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S4', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S5', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
# run_pmmr_experiments(100, 'S6', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))



# MinMax
# run_minmax_experiments(100, 'S1', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S2', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S3', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S4', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S5', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S6', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))


# run_minmax_experiments(100, 'S1', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S2', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S3', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S4', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S5', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
# run_minmax_experiments(100, 'S6', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))


# DeepMMR
# run_mmr_experiments(100, 'S1', 'u', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
run_mmr_experiments(100, 'S1', 'v', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S2', 'u', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S2', 'v', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S3', 'u', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S3', 'v', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S4', 'u', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S4', 'v', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S5', 'u', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S5', 'v', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S6', 'u', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S6', 'v', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))

# run_mmr_experiments(100, 'S1', 'u', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S1', 'v', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S2', 'u', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S2', 'v', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S3', 'u', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S3', 'v', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S4', 'u', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S4', 'v', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S5', 'u', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S5', 'v', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S6', 'u', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
# run_mmr_experiments(100, 'S6', 'v', 4000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))


# RHC
# run_mmr_rhc(100, "u", str(Path.cwd() / 'results' / 'rhc'))
# run_mmr_rhc(100, "v", str(Path.cwd() / 'results' / 'rhc'))