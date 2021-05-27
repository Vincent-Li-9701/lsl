from transformers import LxmertConfig, LxmertModel, LxmertTokenizer
import torch
import torch.nn as nn
from einops import rearrange
PATCH_SIZE = 56

class Lxmert(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_feat_dim, visual_pos_dim, initializer_range, pretrained=False, patch_num = 16, setting='lng_only'):
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
        self.query_out_bn = nn.BatchNorm1d(config.hidden_size, affine=True, track_running_stats=False)
        self.vis_cls = nn.Parameter(torch.zeros(visual_feat_dim))
        self.visual_pos = nn.Embedding(3, visual_pos_dim)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, visual_feats, visual_pos=None, input_ids=None, attention_mask=None, setting=None):
        
        if input_ids is None: # dummpy input sequences
            input_ids = torch.tensor([101, 102]).expand(visual_feats.shape[0], 2).cuda() # Note: expand doesn't copy data
            attention_mask = torch.tensor([0, 0]).expand(visual_feats.shape[0], 2).cuda()
        
        if visual_pos is None: # pre-defined visual positional encoding
            if setting == 'lng_only':
                visual_pos = self.visual_pos(torch.tensor([0, 1]).cuda())
            else:
                visual_pos = self.visual_pos(torch.tensor([0] + [1] * 4 + [2]).cuda())

            visual_pos = visual_pos.expand(visual_feats.shape[0], visual_pos.shape[0], visual_pos.shape[1]).cuda().float()
        
        cls_tokens = self.vis_cls.expand(visual_feats.shape[0], 1, visual_feats.shape[-1])
        visual_feats = torch.cat([cls_tokens, visual_feats], dim=1)

        if self.pretrained:
            out = self.lxmert(input_ids, self.visual_proj(visual_feats), visual_pos, attention_mask=attention_mask).vision_output
        else:
            out = self.lxmert(input_ids, visual_feats, visual_pos, attention_mask=attention_mask).vision_output
        
        concat_out = self.query_out_bn(out[:, 0])
        return self.seq_relationship(concat_out)
    

def init_lxmert(vocab_size, hidden_size, visual_feat_dim, visual_pos_dim):
    config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, 
        visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim)
    return LxmertModel(config)
