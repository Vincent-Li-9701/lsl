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
        self.out_bn = nn.BatchNorm1d(5)
        self.seq_relationship = nn.Linear(config.hidden_size * 2, 2)
        self.dropout = nn.Dropout(p=0.2)
        

    def forward(self, visual_feats, visual_pos=None, input_ids=None, attention_mask=None):
        
        if input_ids is None: # dummpy input sequences
            input_ids = torch.tensor([[101, 102] for _ in range(visual_feats.shape[0])]).reshape((visual_feats.shape[0], -1)).cuda()
            attention_mask = torch.tensor([[0, 0] for _ in range(visual_feats.shape[0])]).reshape((visual_feats.shape[0], -1)).cuda()
        
        if visual_pos is None: # pre-defined visual positional encoding
            visual_pos = torch.tensor([[0,0], [0,1], [0,2], [0,3], [1,0]]).repeat(visual_feats.shape[0], 1, 1).cuda().float()

        if self.pretrained:
            out = self.lxmert(input_ids, self.visual_proj(visual_feats), visual_pos, attention_mask=attention_mask).vision_output
        else:
            out = self.lxmert(input_ids, visual_feats, visual_pos, attention_mask=attention_mask).vision_output
        
        out = self.out_bn(out)
        support_out = torch.squeeze(torch.mean(out[:, :4], dim=1)) # the first 4 outputs of vision encoder corresponds to support image
        query_out = out[:, 4] # the last output of vision encoder corresponds to query image 
        concat_out = torch.cat((support_out, query_out), dim=1)

        return self.dropout(self.seq_relationship(concat_out))
    

def init_lxmert(vocab_size, hidden_size, visual_feat_dim, visual_pos_dim):
    config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, 
        visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim)
    return LxmertModel(config)
