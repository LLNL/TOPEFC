import signac
import numpy as np
import itertools

import platform

def grid(gridspec):
    for values in itertools.product(*gridspec.values()):
        yield dict(zip(gridspec.keys(), values))


plt = platform.system()

workspace = "./"
if plt == "Linux":
    import os
    USER = os.environ['USER']
    workspace = f"/p/lustre2/{USER}/supercapacitor/cap_redox_con_opt/additionals/"
elif plt == "Darwin":
    workspace = "./workspace/"

project = signac.init_project("sprcap-sweep", workspace=workspace)

parameters = {
        "delta": np.array([2e-0]),
        "beta": np.array([0.0, 0.5, 1.0]),
        "lambda": np.array([0.01]),
        "rho_init": np.array([0.5]),
        "mod_brugg": np.array([0.02]),
        "engy_cval": np.array([0.5]),
        "p_init": np.array([1.0]),
        "loop_init": np.array([350]),
        "opt_cycle": np.array([0]),
        "opt_strat": np.array([0]),
        "Nx": np.array([64]),
        }

for sp in grid(parameters):
    job = project.open_job(sp)
    job.init()
