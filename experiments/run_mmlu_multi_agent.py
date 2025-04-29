import asyncio
import argparse
from typing import List, Any, Optional

from swarm.llm import LLMRegistry
from swarm.llm.format import Message
from swarm.graph.swarm import Swarm
from swarm.graph import Node
from swarm.graph import Graph
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.evaluator import Evaluator
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from swarm.environment.agents.agent_registry import AgentRegistry

class CoTStep(Node):
    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 is_last_step: bool,
                 operation_description: str = "Make one step of CoT",
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.is_last_step = is_last_step
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        return self.__class__.__name__

    async def _execute(self, inputs: List[Any] = [], **kwargs):

        node_inputs = self.process_input(inputs)    # 将输入转成list形式
        outputs = []
        for input_dict in node_inputs:

            role = self.prompt_set.get_role()
            constraint = self.prompt_set.get_constraint()
            if self.is_last_step:
                system_prompt = (
                    f"You are {role}. {constraint}. "
                    "Answer taking into consideration the provided sequence "
                    "of thoughts on the question at hand.")
            else:
                system_prompt = (
                    f"You are {role}. "
                    "Given the question, solve it step by step. "
                    "Answer your thoughts about the next step of the solution given "
                    "everything that has been provided to you so far. "
                    "Expand on the next step. "
                    "Do not try to provide the answer straight away, instead expand "
                    "on your thoughts about the next step of the solution."
                    "Aswer in maximum 30 words. "
                    "Do not expect additional input. Make best use of whatever "
                    "knowledge you have been already provided.")
            if 'output' in input_dict:    # 这里判断是否接受上一段的输出
                task = input_dict['output']
            else:
                task = input_dict["task"]
            user_prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt)]
            response = await self.llm.agen(message, max_tokens=50)
            if self.is_last_step:
                concatenated_response = response
            else:
                concatenated_response = f"{task}. Here is the next thought. {response}. "

            execution = {
                "operation": self.node_name,
                "task": task,
                "files": input_dict.get("files", []),
                "input": task,
                "role": role,
                "constraint": constraint,
                "prompt": user_prompt,
                "output": concatenated_response,
                "ground_truth": input_dict.get("GT", []),
                "format": "natural language"
            }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        return outputs


@AgentRegistry.register('CustomCOT')
class CustomCOT(Graph):

    def build_graph(self):

        num_thoughts = 3

        assert num_thoughts >= 2

        thoughts = []
        for i_thought in range(num_thoughts):
            thought = CoTStep(self.domain,
                           self.model_name,
                           is_last_step=i_thought==num_thoughts-1)
            if i_thought > 0:
                thoughts[-1].add_successor(thought)
            thoughts.append(thought)

        self.input_nodes = [thoughts[0]]
        self.output_nodes = [thoughts[-1]]

        for thought in thoughts:
            self.add_node(thought)


async def main():

    # args = parse_args()

    # debug: bool = args.debug

    # model_name: Optional[str] = args.model_name

    # mode: Union[Literal['DirectAnswer'],
    #             Literal['FullConnectedSwarm'],
    #             Literal['RandomSwarm'],
    #             Literal['OptimizedSwarm']]

    # mode = args.mode

    model_name = 'GLM'
    mode = 'OptimizedSwarm'
    strategy = MergingStrategy.MajorityVote

    # domain: str = args.domain
    domain = 'mmlu'

    if mode == 'DirectAnswer':
        swarm_name = None
        swarm = None
    else:
        #N = args.num_truthful_agents
        N = 2
        M = N
        num_thoughts = 2
        agent_name_list = N * ["IO"] + M * ["AdversarialAgent"] + ["CustomCOT"]*2

        swarm_name = f"{N}true_{M}adv_{num_thoughts}cot"

        swarm = Swarm(    # 创建一个swarm，swarm是所有agent的集合
            agent_name_list,
            domain,
            model_name=model_name,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy),
            edge_optimize=True,
        )

    tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}"

    download()

    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')

    evaluator = Evaluator(
        swarm,
        dataset_train,
        dataset_val,
        model_name=model_name,
        enable_tensorboard = mode=='OptimizedSwarm',
        enable_artifacts=True,
        tensorboard_tag=tag)

    limit_questions = 5 # if debug else 153

    if mode == 'DirectAnswer':
        score = await evaluator.evaluate_direct_answer(
            limit_questions=limit_questions)
    elif mode == 'FullConnectedSwarm':
        score = await evaluator.evaluate_swarm(
            mode='full_connected_swarm',
            limit_questions=limit_questions)
    elif mode == 'RandomSwarm':
        score = await evaluator.evaluate_swarm(
            mode='randomly_connected_swarm',
            limit_questions=limit_questions)
    elif mode == 'OptimizedSwarm':

        num_iters = 5 # if debug else args.num_iterations

        lr = 0.1

        edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr)

        score = await evaluator.evaluate_swarm(
            mode='external_edge_probs',
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            )
    else:
        raise Exception(f"Unsupported mode {mode}")

    print(f"Score: {score}")


if __name__ == "__main__":
    asyncio.run(main())
