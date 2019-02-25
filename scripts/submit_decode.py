import itertools
from slurmpy import Slurm

partition = 'gpu_yugroup'

# sweep lambda_reg
params_to_vary = {
    '--reg': [0, 1e-1, 5e-1, 1e0, 1e1, 1e2]
}


# run
s = Slurm("proto", {"partition": partition, "time": "3-0", "gres": "gpu:1"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; module load pytorch; python ../train.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
