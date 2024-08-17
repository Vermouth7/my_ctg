import torch
from transformers import Pipeline
from transformers.pipelines import TextGenerationPipeline

from .rep_control_vec import WrappedReadingVecModel

device = torch.device("cuda")


class RepControlPipeline(TextGenerationPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers=None, 
                 block_name="decoder_block", 
                 **kwargs):
        
        # TODO: implement different control method and supported intermediate modules for different models
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        if layers:
            self.wrapped_model.wrap_block(layers-1, block_name=block_name)
            self.block_name = block_name
            self.layers = layers-1
        

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
   
    def __call__(self, text_inputs, token_pos=-1,activations=None, logits=None, control_method="hidden_states",**kwargs):

        if control_method=="hidden_states" and activations is not None:
            if kwargs.get('batch_size')!=1:
                activations=torch.transpose(activations,0,1)
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name,token_pos)

        
        if control_method=='hidden_states':
            outputs = super().__call__(text_inputs=text_inputs,return_full_text=False, use_cache=False,**kwargs)
            # outputs=self.ctg_hidden_states(text_inputs,kwargs.get('batch_size'),kwargs.get('max_new_tokens'))
        elif control_method=='logits':
            outputs=self.ctg_logits(text_inputs,logits,kwargs.get('batch_size'),kwargs.get('max_new_tokens'))
            
        self.wrapped_model.reset()
        return outputs
    
    def ctg_hidden_states(self,text_inputs,batch_size,max_new_tokens):
        res = []

        encoded_inputs = self.tokenizer(text_inputs, return_tensors="pt",padding=True)
        encoded_inputs.to(device)
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"].to(self.device)
        new_attention = torch.Tensor([1]*batch_size).to(device).unsqueeze(1)
        
        pre_token = input_ids.shape[1]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask, use_cache=False)
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1,keepdim=True)
            
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            attention_mask = torch.cat((attention_mask, new_attention), dim=1)


            if (input_ids.squeeze(-1) == self.tokenizer.eos_token_id).all():
                break

        ps = self.tokenizer.batch_decode(input_ids[:, pre_token:], skip_special_tokens=True)
        res.extend(ps)
        return res
    
    def ctg_logits(self,text_inputs,logits,batch_size,max_new_tokens):
        res=[]
        encoded_inputs = self.tokenizer(text_inputs, return_tensors="pt",padding=True)
        encoded_inputs.to(device)
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"].to(self.device)
        new_attention = torch.Tensor([1]*batch_size).to(device).unsqueeze(1)
        
        pre_token = input_ids.shape[1]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask, use_cache=True)
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits=torch.add(next_token_logits,logits)
            
            next_token_id = torch.argmax(next_token_logits, dim=-1,keepdim=True)
            
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            attention_mask = torch.cat((attention_mask, new_attention), dim=1)


            if (input_ids.squeeze(-1) == self.tokenizer.eos_token_id).all():
                break

        ps = self.tokenizer.batch_decode(input_ids[:, pre_token:], skip_special_tokens=True)
        res.extend(ps)
        return res
    
    def ctg_mix(self,text_inputs,batch_size,logits):
        pass