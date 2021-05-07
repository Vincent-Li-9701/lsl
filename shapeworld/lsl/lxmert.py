from transformers import LxmertConfig, LxmertModel, LxmertTokenizer
import torch
import torch.nn as nn
from einops import rearrange
PATCH_SIZE = 56

class Lxmert(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_feat_dim, visual_pos_dim, initializer_range, pretrained=False, patch_num = 16):
        super().__init__()
        self.pretrained = pretrained
        if self.pretrained:
            self.visual_proj = nn.Linear(visual_feat_dim, 2048)
            self.lxmert = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
        else:
            config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, \
               visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim, initializer_range=initializer_range)
            
            self.lxmert = LxmertModel(config)

        # classification head for matching task
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        

    def forward(self, visual_feats, visual_pos=None, input_ids=None, attention_mask=None):
        
        if input_ids is None:
            input_ids = torch.tensor([[101, 102] for _ in range(visual_feats.shape[0])]).reshape((visual_feats.shape[0], -1)).cuda()
            attention_mask = torch.tensor([[0, 0] for _ in range(visual_feats.shape[0])]).reshape((visual_feats.shape[0], -1)).cuda()
      
        if visual_pos is None:
            if len(visual_feats.shape) > 2:
                visual_pos = torch.tensor([[0,0], [0,1], [0,2], [0,3]]).repeat(visual_feats.shape[0], 1, 1).cuda().float()
            else:
                visual_pos = torch.tensor([1,0]).repeat(visual_feats.shape[0], 1).cuda().float()
                visual_feats = torch.unsqueeze(visual_feats, dim=1)
        
        if self.pretrained:
            out = self.lxmert(input_ids, self.visual_proj(visual_feats), visual_pos, attention_mask=attention_mask).vision_output
        else:
            out = self.lxmert(input_ids, visual_feats, visual_pos, attention_mask=attention_mask).vision_output
        
        # return self.seq_relationship(out)
        out = torch.mean(out, dim=1) # use when using vision_output
        out = nn.functional.normalize(out, dim=-1)
       
        return out
    

def init_lxmert(vocab_size, hidden_size, visual_feat_dim, visual_pos_dim):
    config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, 
        visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim)
    return LxmertModel(config)
