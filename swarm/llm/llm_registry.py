from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"

        if model_name == 'mock':
            model = cls.registry.get('mock')
        elif model_name == 'GLM':
            model = cls.registry.get('GLMChat', model_name='glm-4-flash')
        else: # any version of GPTChat like "gpt-4-1106-preview"
            model = cls.registry.get('GPTChat', model_name=model_name)

        return model
