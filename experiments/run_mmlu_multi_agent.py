import asyncio
import argparse
from typing import List, Any, Optional, Union, Literal

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


def parse_args():
    parser = argparse.ArgumentParser(description='Run MMLU multi-agent evaluation')
    
    # Basic configuration
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--model_name', type=str, default='GLM', help='Name of the model to use')
    parser.add_argument('--domain', type=str, default='mmlu', help='Domain to evaluate on')
    
    # Mode selection
    parser.add_argument('--mode', type=str, 
                       choices=['DirectAnswer', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                       default='OptimizedSwarm', help='Evaluation mode')
    
    # Optimizer configuration
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimization')
    
    # Shapley value optimization
    parser.add_argument('--use_shapley', action='store_true', help='Use Shapley value optimization')
    parser.add_argument('--shapley_samples', type=int, default=20, 
                        help='Number of Monte Carlo samples for Shapley estimation')
    parser.add_argument('--shapley_threshold', type=float, default=0.0, 
                        help='Threshold for including edges based on Shapley values')
    parser.add_argument('--shapley_lr', type=float, default=0.2, 
                        help='Learning rate for Shapley updates')
    parser.add_argument('--visualize_shapley', action='store_true', 
                        help='Generate Shapley value visualizations')
    parser.add_argument('--shapley_max_edges', type=int, default=50,
                        help='Maximum number of edges to evaluate for Shapley values')
    parser.add_argument('--shapley_time_budget', type=int, default=600,
                        help='Time budget for Shapley computation in seconds')
    parser.add_argument('--shapley_parallel', action='store_true', default=True,
                        help='Use parallel computation for Shapley values')
    parser.add_argument('--shapley_batch_size', type=int, default=5,
                        help='Batch size for parallel Shapley computation')
    
    # Swarm configuration
    parser.add_argument('--num_truthful_agents', type=int, default=2, 
                        help='Number of truthful agents in the swarm')
    parser.add_argument('--num_adversarial_agents', type=int, default=2, 
                        help='Number of adversarial agents in the swarm')
    parser.add_argument('--num_cot_agents', type=int, default=2, 
                        help='Number of CoT agents in the swarm')
    
    # Evaluation configuration
    parser.add_argument('--limit_questions', type=int, default=5, 
                        help='Maximum number of questions to evaluate')
    
    return parser.parse_args()

async def main():

    # args = parse_args()

    # debug: bool = args.debug

    # model_name: Optional[str] = args.model_name

    # mode: Union[Literal['DirectAnswer'],
    #             Literal['FullConnectedSwarm'],
    #             Literal['RandomSwarm'],
    #             Literal['OptimizedSwarm']] = args.mode

    # # Shapley value optimization parameters
    # use_shapley = args.use_shapley
    # shapley_samples = args.shapley_samples
    # shapley_threshold = args.shapley_threshold  
    # shapley_lr = args.shapley_lr
    # visualize_shapley = args.visualize_shapley

    # domain: str = args.domain

    model_name = "GLM"
    domain = "mmlu"
    mode = "OptimizedSwarm"
    use_shapley = True
    shapley_samples = 2
    shapley_threshold = 0.4
    shapley_lr = 0.2
    visualize_shapley = True
    shapley_max_edges = 10
    shapley_time_budget = 600
    shapley_parallel = True
    shapley_batch_size = 5
    if mode == 'DirectAnswer':
        swarm_name = None
        swarm = None
    else:
        N = 2 # args.num_truthful_agents
        M = 2 # args.num_adversarial_agents
        num_thoughts = 0 # args.num_cot_agents
        strategy = MergingStrategy.MajorityVote
        
        agent_name_list = N * ["IO"] + M * ["AdversarialAgent"] + ["CustomCOT"]*num_thoughts

        swarm_name = f"{N}true_{M}adv_{num_thoughts}cot"

        swarm = Swarm(    # 创建一个swarm，swarm是所有agent的集合
            agent_name_list,
            domain,
            model_name=model_name,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy),
            edge_optimize=True,
        )

    # Add Shapley to tag if enabled
    if use_shapley and mode == 'OptimizedSwarm':
        tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_Shapley"
    else:
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

    limit_questions = 5 # args.limit_questions if not debug else 5

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

        num_iters = 2 # args.num_iterations if not debug else 5
        lr = 0.1 # args.lr

        edge_probs = await evaluator.optimize_swarm(
            num_iters=num_iters, 
            lr=lr,
            use_shapley=use_shapley,
            shapley_samples=shapley_samples,
            shapley_threshold=shapley_threshold,
            shapley_lr=shapley_lr,
            visualize_shapley=visualize_shapley,
            shapley_max_edges=shapley_max_edges,
            shapley_time_budget=shapley_time_budget,
            shapley_parallel=shapley_parallel,
            shapley_batch_size=shapley_batch_size
        )

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
