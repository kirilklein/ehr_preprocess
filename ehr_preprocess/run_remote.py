import sys
from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig

ws = Workspace.from_config()

job_name = sys.argv[1]

env = Environment.get(ws, name="Aiomic-Deploy")

ct = "CPU-20-LP"
src = ScriptRunConfig(source_directory=".", script=job_name+".py", compute_target=ct, environment=env)
Experiment(ws, name="Format_EHR").submit(src)