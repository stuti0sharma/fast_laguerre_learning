"""
Script for submitting Python jobs to cluster environments.
This script supports job submission to different clusters, handling environment setup,
output directories, and batch script generation.
"""

import subprocess
import argparse
import socket
import os
from os.path import join, dirname, realpath, exists, expandvars
import time
from shutil import copy, copytree

parser = argparse.ArgumentParser(description="Submit script for python scripts")

parser.add_argument("--f", action="store", dest="file", default=None, type=str)
parser.add_argument("--n", action="store", dest="n_jobs", default=1, type=int)
parser.add_argument("--m", action="store", dest="name", default="", type=str)
parser.add_argument("--nodes", action="store", dest="nodes", default="1", type=str)
parser.add_argument("--cpus-per-task", action="store", dest="cpus", default="16", type=str)
parser.add_argument("--ntasks-per-node", action="store", dest="tasks", default="1", type=str)
parser.add_argument("--time", action="store", dest="time", default="24:00:00", type=str)
parser.add_argument("--sub", action="store", dest="subfolder", default="", type=str,
                    help="subfolder name to store the submitted outputs in. Usually set for scans (i.e., n > 1)")

parser.add_argument(
    "--gtype",
    action="store",
    dest="gpu_type",
    default="a40",
    type=str,
    help="Possbile options on NHR Alex (Erlangen cluster) are, 'a40', 'a100', 'a100_80'",
)
parser.add_argument("--g", action="store", dest="gpus", default="1", type=str)
parser.add_argument("--run-copy", action="store", dest="run_copy", default="True", type=bool)
parser.add_argument("--sleep", action="store", dest="sleep", default="1.", type=float)

parsed = parser.parse_args()

assert parsed.file is not None, "No file for submission given. Add it via 'python submit.py --f $FILE'"
assert ".py" in parsed.file, "Submitted file has to be a pyton file"
assert os.path.isfile(parsed.file) is True, "Cannot locate file %s" % parsed.file

work_dir = expandvars("$WORK")
env_dir = expandvars("$CONDA_DEFAULT_ENV")

if env_dir == "base":
    env_dir = expandvars("$LAGUERREENV")

if "laguerre_learning" not in env_dir:
    print("\n - - - - - - - - - - - - - - - - - !!!! WARNING: !!!! - - - - - - - - - - - - - - - - - ")
    print("It looks like you did not activate your laguerre_learning conda environment.")
    print("Either activate your environment (for default installation: 'conda activate laguerre_learning'")
    print("Or declare your env as $LAGUERREENV variable in your .bashrc")
    print(" - - - - - - - - - - - - - - - - - !!!! WARNING: !!!! - - - - - - - - - - - - - - - - - \n")

if parsed.name == "":
    file_name = parsed.file.split("/")[-1].split(".py")[0]
else:
    file_name = parsed.name

project_root = dirname(realpath(__file__))
relative_file_dir = os.path.relpath(dirname(realpath(parsed.file)), project_root)
folder = relative_file_dir if relative_file_dir != "." else "root"

out_dir = str(join(work_dir, "jobs", folder))
os.makedirs(out_dir, exist_ok=True)

subfolder = parsed.subfolder

if subfolder != "":
    job_folder_dir = join(out_dir, subfolder)
elif parsed.n_jobs > 1:
    job_folder_dir = out_dir
    scan_id = 0
    while exists(join(job_folder_dir, "%s_scan_%i" % (file_name, scan_id))):
        scan_id += 1
    job_folder_dir = join(job_folder_dir, "%s_scan_%i" % (file_name, scan_id))
else:
    job_folder_dir = out_dir

os.makedirs(job_folder_dir, exist_ok=True)

file_path = realpath(parsed.file)
job_folder_dir = join(job_folder_dir, file_name)

for i in range(parsed.n_jobs):
    run_id = 0
    while exists(job_folder_dir + "_job_%i" % run_id):
        run_id += 1

    job_dir = job_folder_dir + "_job_%i" % run_id

    file_name_sh = file_name + "_job_%i" % run_id
    print("Creating folder.....", job_dir)
    os.makedirs(job_dir)
    model_dir = join(job_dir, "backup", "models")
    os.makedirs(model_dir)

    copied_file = copy(file_path, job_dir)


    batch_file = join(job_dir, file_name_sh + ".sh")

    if parsed.run_copy is True:
        file_path = copied_file

    if parsed.gpu_type not in ["a40", "a100", "a100_80"]:
        raise KeyError('The gpu_type (--gtype) has to be one of "a40", "a100", "a100_80"')

    with open(batch_file, "w") as f:
        f.writelines("#!/bin/bash -l\n")
        f.writelines("\n#SBATCH --nodes=%s" % parsed.nodes)
        f.writelines("\n#SBATCH --ntasks-per-node=%s" % parsed.tasks)
        f.writelines("\n#SBATCH --cpus-per-task=%s" % parsed.cpus)
        f.writelines("\n#SBATCH --time=%s" % parsed.time)

        if parsed.gpu_type == "a100_80":
            f.writelines("\n#SBATCH --gres=gpu:a100:%s -C a100_80" % parsed.gpus)
        else:
            f.writelines("\n#SBATCH --gres=gpu:%s:%s" % (parsed.gpu_type, parsed.gpus))

        f.writelines("\n#SBATCH --output=output_test.out")
        f.writelines("\n\nconda activate %s\n" % env_dir)
        f.writelines("\npython %s --log_dir %s" % (file_path, job_dir))


    time.sleep(0.1)
    process = subprocess.run(["cd %s && sbatch ./%s.sh" % (job_dir, file_name_sh)], capture_output=True, shell=True)
    print(process.stdout.decode())
    time.sleep(parsed.sleep)
