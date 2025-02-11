from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import openai

class OpenLLMAPI(LLM):
    import openai
    client: openai.Client
    model: str

    @property
    def _llm_type(self) -> str:
        return "OpenLLMAPI"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
    ) -> str:
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 4000
        if 'n' in kwargs and kwargs['n'] != 1:
            kwargs['n'] = 1
            print('Warning: resetting n=1')

        temperature = 0
        while(temperature < 0.5):
            try:
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        }
                    ],
                    stop=stop,
                    temperature=temperature,
                    **kwargs,
                )
                ret = result.choices[0].message.content.strip()
                temperature = 1
            except:
                temperature +=0.2
                ret = "error"

        return ret

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"client": self.client, "model": self.model}


def create_open_llm(url):
    client = openai.Client(
        api_key="empty",
        base_url=url)
    models = client.models.list()
    model = models.data[0].id
    print('url:', url)
    print('model:', model)
    return OpenLLMAPI(
        client=client,
        model=model)

