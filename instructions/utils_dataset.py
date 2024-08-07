import json
import os
import random
import time
from http import HTTPStatus

import huggingface_hub
import pandas as pd
import requests
from dashscope import Generation
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer


def query_qwen_dataset(prompt):
    messages = [{'role': 'system', 'content': 'You are an intelligent writing assistant.'},
                {'role': 'user', 'content': prompt}]
    gen = Generation()

    max_tries = 15
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            response = gen.call(
                'qwen-plus',
                messages=messages,
                result_format='message'
            )
            if response.status_code == HTTPStatus.OK:
                response = response['output']['choices'][0]['message']['content'].strip()
                success = True
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                success = False
        except Exception as e:
            print(f"Error encountered: {e}")
            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(10)
    return response