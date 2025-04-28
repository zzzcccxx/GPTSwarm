#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from typing import List, Any, Optional
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer


class DirectAnswer(Node): 
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Directly output an answer.",
                 max_token: int = 50, 
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()


    @property
    def node_name(self):
        return self.__class__.__name__
    
    async def node_optimize(self, input, meta_optmize=True):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optmize:
            update_role = role 
            node_optmizer = MetaPromptOptimizer(self.domain, self.model_name)
            # Create a prompt for the question
            prompt = self.prompt_set.get_answer_prompt(question=task)
            
            # Create test case for the meta_evaluator
            expected_output = input.get("GT", None)
            if expected_output is None:
                logger.warning("No ground truth answer provided. Using sample answer for testing.")
                expected_output = "Sample answer for testing"
            
            test_case = {
                "test_name": "question_test",
                "test_input": task,
                "expected_output": expected_output,
                "validation_type": "contains"  # Just check if output contains expected string
            }
            
            # Call generate with all required parameters
            update_constraint = await node_optmizer.generate(
                init_prompt=prompt,
                init_constraint=constraint,
                init_role=role,
                tests=test_case,
                data_desc="question answering"
            )
            return update_role, update_constraint

        return role, constraint


    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []

        for input in node_inputs:
            task = input["task"]
            role, constraint = await self.node_optimize(input, meta_optmize=True)
            prompt = self.prompt_set.get_answer_prompt(question=task)    
            message = [Message(role="system", content=f"You are a {role}. {constraint}"),
                       Message(role="user", content=prompt)]
            response = await self.llm.agen(message, max_tokens=self.max_token)

            execution = {
                "operation": self.node_name,
                "task": task,
                "files": input.get("files", []),
                "input": task,
                "role": role,
                "constraint": constraint,
                "prompt": prompt,
                "output": response,
                "ground_truth": input.get("GT", []),
                "format": "natural language"
            }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        # self.log()
        return outputs 