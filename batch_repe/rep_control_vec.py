# wrapping classes
import numpy as np
import torch


class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.normalize = False
        self.input_pos=None
        
        
    def forward(self,*args,edit=False, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output
        # print("output 0:")
        # print(output[0].shape)
        # print(self.controller.shape)
        if self.controller is not None and edit == True:
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

            if self.mask is not None:
                mask = self.mask
            
            # we should ignore the padding tokens when doing the activation addition
            # mask has ones for non padding tokens and zeros at padding tokens.
            # only tested this on left padding
            elif "position_ids" in kwargs:
                pos = kwargs["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                mask = mask.to(modified.dtype)
            else:
                # print(f"Warning: block {self.block_name} does not contain information 'position_ids' about token types. When using batches this can lead to unexpected results.")
                mask = 1.0
            # print("mask",mask.shape)
            
            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            # assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."
            
            self.controller = self.controller.to(modified.device)
            
            if type(mask) == torch.Tensor:
                mask = mask.to(modified.device)
            # handle activation
            # print(self.token_pos)
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller[:, self.input_pos] * mask[:, self.input_pos])
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple):
                for i in range(0,len(self.token_pos)):
                    token=self.token_pos[i]
                    modified[:, token] = self.operator(modified[:, token], self.controller[self.input_pos[i], -1].unsqueeze(0) * mask[:, -1])
            if self.normalize:
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre
            # if isinstance(self.token_pos,int):
            #     self.token_pos-=1
            # elif isinstance(self.token_pos,list):
            #     self.token_pos = [pos - 1 for pos in self.token_pos]
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False, operator='replace'):
        self.normalize = normalize
        self.controller = activations
        self.mask = masks
        # self.token_pos=token_pos
        # self.input_pos=token_pos
        if operator == 'linear_comb':
            def op(current, controller):
                return current + controller
        elif operator == 'piecewise_linear':
            def op(current, controller):
                sign = torch.sign((current * controller).sum(-1, keepdim=True))
                return current + controller * sign
        elif operator == 'projection':
            def op(current, controller):
                raise NotImplementedError
        elif operator == 'replace':
            def op(current,controller):
                return controller
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")
        self.operator = op
        
    def reset(self):
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.operator = None

    def set_masks(self, masks):
        self.mask = masks
    def set_token_pos(self,token_pos):
        if isinstance(token_pos,list):
            self.input_pos=[-1]*len(token_pos)
        else:
            self.input_pos=-1
        self.token_pos=token_pos

BLOCK_NAMES = [
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm"
    ]
    
class WrappedReadingVecModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
        
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits
        
    def run_prompt(self, prompt, **kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output

    def wrap(self, layer_id, block_name):
        assert block_name in BLOCK_NAMES
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = getattr(self.model.model.layers[layer_id].block, block_name)
            if not self.is_wrapped(block):
                setattr(self.model.model.layers[layer_id].block, block_name, WrappedBlock(block))
        else:
            block = getattr(self.model.model.layers[layer_id], block_name)
            if not self.is_wrapped(block):
                setattr(self.model.model.layers[layer_id], block_name, WrappedBlock(block))

    def wrap_decoder_block(self, layer_id):
        block = self.model.model.layers[layer_id]
        if not self.is_wrapped(block):
            self.model.model.layers[layer_id] = WrappedBlock(block)

    def wrap_all(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            # for block_name in BLOCK_NAMES:
            #     self.wrap(layer_id, block_name)
            self.wrap_decoder_block(layer_id)
            
    def wrap_block(self, layer_ids, block_name):
        def _wrap_block(layer_id, block_name):
            if block_name in BLOCK_NAMES:
                self.wrap(layer_id, block_name)
            elif block_name == 'decoder_block':
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)

    def get_activations(self, layer_ids, block_name='decoder_block'):

        def _get_activations(layer_id, block_name):
            current_layer = self.model.model.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'decoder_block':
                    return current_layer.output
                elif block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_block, block_name)):
                    return getattr(current_block, block_name).output
                else:
                    assert False, f"No wrapped block named {block_name}."

            else:
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_layer, block_name)):
                    return getattr(current_layer, block_name).output
                else:
                    assert False, f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            activations = {}
            for layer_id in layer_ids:
                activations[layer_id] = _get_activations(layer_id, block_name)
            return activations
        else:
            return _get_activations(layer_ids, block_name)


    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=-1, masks=None, normalize=False, operator='replace'):
        
        def _set_controller(layer_id, activations, block_name, masks, normalize, operator):
            current_layer = self.model.model.layers[layer_id]
            if block_name == 'decoder_block':
                current_layer.set_controller(activations, token_pos, masks, normalize, operator)
            elif self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_block, block_name)):
                    getattr(current_block, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."

            else:
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_layer, block_name)):
                    getattr(current_layer, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[:,layer_id+1], block_name, masks, normalize, operator)
        elif isinstance(layer_ids,int):
            _set_controller(layer_ids, activations[:,layer_ids+1], block_name, masks, normalize, operator)
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize, operator)
      
        
    def reset(self):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.reset()
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer.block, block_name)):
                        getattr(layer.block, block_name).reset()
            else:
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer, block_name)):
                        getattr(layer, block_name).reset()

    def set_masks(self, masks):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.set_masks(masks)
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer.block, block_name)):
                        getattr(layer.block, block_name).set_masks(masks)
            else:
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer, block_name)):
                        getattr(layer, block_name).set_masks(masks)

    def is_wrapped(self, block):
        if hasattr(block, 'block'):
            return True
        return False
    
    def unwrap(self):
        for l, layer in enumerate(self.model.model.layers):
            if self.is_wrapped(layer):
                self.model.model.layers[l] = layer.block
            for block_name in BLOCK_NAMES:
                if self.is_wrapped(getattr(self.model.model.layers[l], block_name)):
                    setattr(self.model.model.layers[l],
                            block_name,
                            getattr(self.model.model.layers[l], block_name).block)
