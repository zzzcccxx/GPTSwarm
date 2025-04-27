import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from swarm.graph.swarm import Swarm

swarm = Swarm(["IO", "IO", "IO"], "gaia", model_name='GLM')
task = "What is the capital of Jordan?"
inputs = {"task": task}
answer = swarm.run(inputs)
print(answer)