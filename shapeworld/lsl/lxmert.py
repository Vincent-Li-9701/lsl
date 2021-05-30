from transformers import LxmertConfig, LxmertModel, LxmertTokenizer
import torch
import torch.nn as nn
from einops import rearrange
PATCH_SIZE = 56

class Lxmert(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_feat_dim, visual_pos_dim, initializer_range, pretrained=False, patch_num = 16, setting='lng_only'):
        super().__init__()
        
        config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, \
            visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim, initializer_range=initializer_range)
        self.lxmert = LxmertModel(config)

        # classification head for matching task
        self.query_out_bn = nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=False)
        self.support_out_bn = nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=False)
        self.caption_out_bn = nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=False)
        self.v2v_head = nn.Bilinear(hidden_size, hidden_size, 1)
        self.v2l_head = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, visual_feats, visual_pos=None, input_ids=None, attention_mask=None, setting=None):
        
        if input_ids is None: # dummpy input sequences
            input_ids = torch.tensor([101, 102]).expand(visual_feats.shape[0], 2).cuda() # Note: expand doesn't copy data
            attention_mask = torch.tensor([0, 0]).expand(visual_feats.shape[0], 2).cuda()
        
        if visual_pos is None: # pre-defined visual positional encoding
            if setting == 'lng_only':
                visual_pos = [[1,0]]
            else:
                visual_pos = [[0,0], [0,1], [0,2], [0,3], [1,0]]

            visual_pos = torch.tensor(visual_pos).repeat(visual_feats.shape[0], 1, 1).cuda().float()

        out = self.lxmert(input_ids, visual_feats, visual_pos, attention_mask=attention_mask)
        v_out, l_out = out.vision_output, out.language_output
        
        q_out = self.query_out_bn(v_out[:, -1])
        lq_score, sq_score = None, None
        if setting != 'meta':
            l_out = self.caption_out_bn(l_out[:,0])
            lq_score = self.v2l_head(l_out, q_out)
        
        if setting != 'lng_only':
            s_out = torch.squeeze(torch.mean(v_out[:, :4], dim=1)) # the first 4 outputs of vision encoder corresponds to support image
            s_out = self.support_out_bn(s_out)
            sq_score = self.v2v_head(s_out, q_out)
        
        return sq_score, lq_score
    

def init_lxmert(vocab_size, hidden_size, visual_feat_dim, visual_pos_dim):
    config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, 
        visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim)
    return LxmertModel(config)
