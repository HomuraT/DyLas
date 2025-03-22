import openai
import os
from openai import OpenAI
from zhipuai import ZhipuAI
import requests
import json

OpenAI_config = {
    'temperature': 0
}

class OpenAI_API:
    def __init__(self, url, model, api_key=None, history=None, max_tokens=None):
        self.url = url
        self.api_key = api_key
        self.model = model

        openai.api_base = url
        openai.api_key = api_key if api_key else 'none'
        self.temperature = OpenAI_config['temperature']
        self.max_tokens = max_tokens

        if 'glm' in model:
            if 'chatglm' in model:
                self.client = ZhipuAI(api_key=self.api_key, base_url=url)
            elif 'glm-4-9b' in model:
                self.model = 'glm-4'
                self.client = OpenAI(api_key=self.api_key, base_url=url)
            else:
                self.client = ZhipuAI(api_key=self.api_key)
            if self.temperature == 0:
                self.temperature = 0.01
        elif self.api_key == 'ollama':
            self.client = OpenAI(
                api_key=openai.api_key,
                base_url=url
            )
        else:
            self.client = OpenAI(
                api_key=openai.api_key,
            )

        self.history = history if history else []

    def send_messages(self, messages):
        openai.api_base = self.url
        openai.api_key = self.api_key

        if openai.api_key == 'ollama':
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "use_mmap": False
                }
            }

            response = requests.post(self.url, headers=headers, data=json.dumps(data))

            return response.json()

        return self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)

    def _get_result_content(self, response_by_api):
        if openai.api_key == 'ollama':
            return response_by_api['message']['content']

        return response_by_api.choices[-1].message.content

    def chat(self, text):
        message = {"role": "user", "content": text}
        self.history.append(message)

        result = self.send_messages(self.history)

        result_content = self._get_result_content(result)

        self.history.append({"role": "assistant", "content": result_content})

        return result_content

    def chat_without_history(self, text):
        message = [{"role": "user", "content": text}]
        result = self.send_messages(message)
        result_content = self._get_result_content(result)

        return result_content

    def to_dict(self):
        return {
            'url': self.url,
            'api_key': self.api_key,
            'model': self.model,
            'history': self.history,
        }


class API_Manager:
    def __init__(self, save_path=None):
        '''
        用来管理不同的历史记录
        '''
        self.apis = {}
        self.save_path = save_path

    def add_api(self, id, api: OpenAI_API):
        if id in self.apis:
            return False
        else:
            self.apis[id] = api
            return True

    def set_save_path(self, path):
        self.save_path = path

    def save(self):
        apis_status = {}
        for id, api in self.apis.items():
            apis_status[id] = api.to_dict()

        assert self.save_path is not None, 'save path is None'
        json.dump(apis_status, open(self.save_path, 'w', encoding='utf-8'), indent=5)
        return self

    def load(self):
        assert self.save_path is not None, 'save path is None'
        apis_status = json.load(open(self.save_path, 'r', encoding='utf-8'))
        for id, api_stat in apis_status.items():
            self.apis[id] = OpenAI_API(**api_stat)
        return self

    def get_api_by_id(self, id):
        if id not in self.apis:
            return None
        return self.apis[id]
