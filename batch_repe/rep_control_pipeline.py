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
        assert isinstance(layers,list),"layers layers be list"
        layers=[i-1 for i in layers]
        if isinstance(layers, list):
            # self.wrapped_model.wrap_block(layers, block_name=block_name)
            self.wrapped_model.wrap_all()
            self.block_name = block_name
            self.layers = layers
        

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
   
    def __call__(self, text_inputs, tokenizer,token_pos=-1,activations=None, logits=None, control_method="hidden_states",**kwargs):

        if control_method=="hidden_states" and activations is not None and self.layers is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name,token_pos)
        self.set_pos(text_inputs,tokenizer)

        if control_method=='hidden_states':
            outputs = super().__call__(text_inputs=text_inputs,return_full_text=False, use_cache=False,output_hidden_states=True,**kwargs)
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
    def set_pos(self,inputs,tokenizer):
        input_ids=tokenizer(inputs,padding=True,truncation=True,return_tensors="pt").input_ids
        batch,seq_len=input_ids.shape
        token_positions_list=[seq_len-1]*batch
        for layer in self.wrapped_model.model.model.layers:
            layer.set_token_pos(token_positions_list) 
    def ctg_mix(self,text_inputs,batch_size,logits):
        pass
    def get_next_token(self,input_ids,token_pos=-1,activations=None,**kwargs):
        if  activations is not None and self.layers is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name,token_pos)
        else:
            return None
        
        return self.wrapped_model(
            input_ids=input_ids,
        ).logits[:,-1].flatten()