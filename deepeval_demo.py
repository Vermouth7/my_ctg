import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
from deepeval.benchmarks import GSM8K
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda")

class Llama3(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"

model = AutoModelForCausalLM.from_pretrained("/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct",torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct")

Llama3_8b = Llama3(model=model, tokenizer=tokenizer)

# Define benchmark with n_problems and shots
benchmark = GSM8K(
    n_shots=3,
    enable_cot=True
)

# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=Llama3_8b)
print(benchmark.overall_score)