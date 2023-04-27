import json
from langchain.llms.base import LLM
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain import SagemakerEndpoint

from typing import Dict


class ContentHandler(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input = {"ask": prompt, **model_kwargs}
        input_str = json.dumps(input)
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["answer"]

content_handler = ContentHandler()

class SagemakerLLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    llm_type: str = 'ChatGLM'
    # history = []
    model: object = None
    history_len: int = 10
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    def __init__(self):
        super().__init__()

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print('__call__ called')
        return ""

    @property
    def _llm_type(self) -> str:
        return self.llm_type

    def load_model(self, model_name: str, **kwargs):
        chat_model = SagemakerEndpoint(
            endpoint_name=model_name, 
        #         credentials_profile_name="credentials-profile-name", 
            region_name="us-east-1", 
            model_kwargs=kwargs, #{"temperature": 0.1},
            content_handler=content_handler
        )
        self.model = chat_model
