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
               visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim, initializer_range=initializer_range, r_layers=9, x_layers=9)
            
            self.lxmert = LxmertModel(config)
        self.score_proj = nn.Linear(hidden_size, 1)
         

    def gen_visual_pos(self, batch_size, num_visual_features, offset=0, test=0):
        grid = int(num_visual_features ** (1/2))
        visual_pos = [[[i/grid + offset, j/grid + offset, 1/grid, 1/grid, test] for i in range(grid) for j in range(grid)] for _ in range(batch_size)]
        return torch.tensor(visual_pos).to(self.lxmert.device)

    def forward(self, support_visual_feats, test_visual_feats, visual_pos=None, input_ids=None, attention_mask=None, use_visual=True):
        original_shape = None
        visual_attention_mask=None
        ''' 
        if len(visual_feats.shape) > 4:
            original_shape = visual_feats.shape
            visual_patches = rearrange(visual_feats, 'b k c (h p1) (w p2) -> (b k) (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        elif len(visual_feats.shape) == 4:
            visual_patches = rearrange(visual_feats, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        if input_ids != None:
            if len(visual_feats.shape) > 4:
                input_ids = torch.repeat_interleave(input_ids, original_shape[1], dim=0)
                attention_mask = torch.repeat_interleave(attention_mask, original_shape[1], dim=0)
        else:
            input_ids = torch.tensor([[101, 102] for _ in range(visual_patches.shape[0])]).reshape((visual_patches.shape[0], -1)).cuda()
        
        if visual_pos is None:
            visual_pos = self.gen_visual_pos(visual_patches.shape[0], visual_patches.shape[1])
       
        if not use_visual:
            visual_attention_mask = torch.zeros((visual_patches.shape[0], visual_patches.shape[1])).cuda()
    
        if self.pretrained:
            out = self.lxmert(input_ids, self.visual_proj(visual_patches), visual_pos, attention_mask=attention_mask, visual_attention_mask=visual_attention_mask).pooled_output
        else:
            out = self.lxmert(input_ids, visual_patches, visual_pos, attention_mask=attention_mask, visual_attention_mask=visual_attention_mask).pooled_output
        #out = torch.mean(out, dim=1)
        
        out = nn.functional.normalize(out, dim=-1)
        if original_shape:
            return out.reshape(*original_shape[:2], -1)
        else:
            return out
        '''
        original_shape = support_visual_feats.shape
        support_visual_patches = rearrange(support_visual_feats, 'b k c (h p1) (w p2) -> b  (k h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        test_visual_patches = rearrange(test_visual_feats, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        visual_patches = torch.cat((test_visual_patches, support_visual_patches), dim=1)
        
        for i in range(original_shape[1]):
            if i == 0:
                support_pos = self.gen_visual_pos(test_visual_patches.shape[0], test_visual_patches.shape[1])
            else:
                support_pos = torch.cat((support_pos, self.gen_visual_pos(test_visual_patches.shape[0], test_visual_patches.shape[1], offset=i)), dim=1)
        test_pos = self.gen_visual_pos(test_visual_patches.shape[0], test_visual_patches.shape[1], offset=original_shape[1], test=1)
        visual_pos = torch.cat((test_pos, support_pos), dim=1)
        output = self.lxmert(input_ids, visual_patches, visual_pos, attention_mask=attention_mask).pooled_output
        score = self.score_proj(output)
        return score

def init_lxmert(vocab_size, hidden_size, visual_feat_dim, visual_pos_dim):
    config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, 
        visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim)
    return LxmertModel(config)
