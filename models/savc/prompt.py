import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self,  args ):
        super().__init__()


        self.pool_size = 10
        self.top_k = 4


        # prompt 使用初始化为 20 * 16 * 56 * 56
        prompt_pool_shape = (self.pool_size, 16, 56, 56)
        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
        nn.init.uniform_(self.prompt, -1, 1)
        
        
        # prompt keys 用模型的分类器进行设置 20 * 512
        key_shape = (self.pool_size, 512)
        self.prompt_key = nn.Parameter(torch.randn(key_shape))
        nn.init.uniform_(self.prompt_key, -1, 1)
            
        
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    
    
    def forward(self, x_embed, data):
        out = dict()

        prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C,     20 * 512
        x_embed_norm = self.l2_normalize(x_embed, dim=1)        # B, C              100 * 512


        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size     100 * 20
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k             100 * 4


        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
        # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
        # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
        # Unless dimension is specified, this will be flattend if it is not already 1D.
        if prompt_id.shape[0] < self.pool_size:
            prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
            id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            
        _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
        major_prompt_id = prompt_id[major_idx] # top_k
        # expand to batch
        idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k


        batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
        batch_size, top_k, length, c1, c2 = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c1, c2) # B, top_k * length, C


        out['prompt_idx'] = idx

        # Debugging, return sim as well
        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity

        # Put pull_constraint loss calculation inside
        batched_key_norm = prompt_norm[idx] # B, top_k, C
        out['selected_key'] = batched_key_norm
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

        out['reduce_sim'] = reduce_sim
    
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        
        normalized_prompt = self.l2_normalize(batched_prompt, dim=1)
        # 展平 batch_size 和 channels 两个维度为一个维度
        flattened_prompt = normalized_prompt.view(-1, c1, c2)
        flattened_data = data.view(-1, c1, c2)

        # 使用 torch.bmm 进行批量点积
        result = torch.bmm(flattened_prompt, flattened_data)

        
        out['prompted_data'] = result.view(batch_size, top_k * length, c1, c2)

        return out
