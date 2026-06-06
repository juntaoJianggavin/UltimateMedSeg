import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, ifsigmoid=False):
        super().__init__()
        self.num_layers = num_layers
        self.ifsigmoid = ifsigmoid
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.ifsigmoid:
            return torch.sigmoid(x)
        else:
            return x
        



def init_bbox_head():
    bbox_decoder = MLP(4096, 512, 4, 3, ifsigmoid=True)    
    nn.init.constant_(bbox_decoder.layers[-1].weight.data, 0)
    nn.init.constant_(bbox_decoder.layers[-1].bias.data, 0)
    return bbox_decoder






class EmbeddingFusion(nn.Module):
    def __init__(self, input_dim=4096):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),  
            nn.GELU(),  
            nn.Linear(1024, 2) 
        )
        
    def forward(self, embed1, embed2):

        fusion_input = torch.cat([embed1, embed2], dim=-1) 
        logits = self.gate(fusion_input)  
        weights = torch.sigmoid(logits) 
        weights = weights / weights.sum(dim=-1, keepdim=True)  
        fused_embedding = weights[:, 0].unsqueeze(-1) * embed1 + weights[:, 1].unsqueeze(-1) * embed2

        return fused_embedding  
    
    
    
    
def init_router():
    router_seg = EmbeddingFusion(4096)
    router_box = EmbeddingFusion(4096)
    
    return router_seg, router_box