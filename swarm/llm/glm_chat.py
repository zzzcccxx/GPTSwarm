import asyncio
import os
from dataclasses import asdict
from typing import List, Union, Optional
from dotenv import load_dotenv
import random
import async_timeout
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time
from typing import Dict, Any

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.price import cost_count
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry

# GLM API base URL
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/"

load_dotenv()
GLM_API_KEYS = os.getenv("GLM_API_KEY")

def glm_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return ''

    client = OpenAI(
        api_key=GLM_API_KEYS,
        base_url=GLM_API_URL
    )

    formated_messages = [asdict(message) for message in messages]
    response = client.chat.completions.create(
        model=model,
        messages=formated_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps
    )
    
    if num_comps == 1:
        cost_count(response, model)
        return response.choices[0].message.content

    cost_count(response, model)

    return [choice.message.content for choice in response.choices]


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def glm_achat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return '' 

    aclient = AsyncOpenAI(
        api_key=GLM_API_KEYS,
        base_url=GLM_API_URL
    )

    formated_messages = [asdict(message) for message in messages]
    try:
        async with async_timeout.timeout(1000):
            response = await aclient.chat.completions.create(
                model=model,
                messages=formated_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=num_comps
            )
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("GLM Timeout")
    if num_comps == 1:
        cost_count(response, model)
        return response.choices[0].message.content
    
    cost_count(response, model)

    return [choice.message.content for choice in response.choices]


@LLMRegistry.register('GLMChat')
class GLMChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return await glm_achat(self.model_name,
                               messages,
                               max_tokens,
                               temperature,
                               num_comps)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return glm_chat(self.model_name,
                        messages, 
                        max_tokens,
                        temperature,
                        num_comps) 