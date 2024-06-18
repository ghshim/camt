import torch
import torch.nn as nn
import math

'''
TODO
- torch.nn.parameter learnable 로 바꿔보기
- start, stop token 바꾸기
- teacher / student learning
'''

class PositionalEncoding(nn.Module):
    '''
    Positional Encoding using sine and cosine
    '''
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # Encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout_p)
    
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Linear(dim_model * 4, dim_model),
            nn.Dropout(dropout_p)
        )
        
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_pad_mask=None):
        # Self-Attention
        src_, _ = self.self_attention(src, src, src, 
                                      attn_mask=src_mask,
                                      key_padding_mask=src_pad_mask)
        src = src + self.dropout(src_)
        src = self.norm1(src)
        # print(f"[Encoder] after self attention, src.shape: {src.shape}")

        # Feed Forward
        src_ = self.feed_forward(src)
        src = src + self.dropout(src_)
        src = self.norm2(src)
        # print(f"[Encoder] after feed forward network, src.shape: {src.shape}")

        return src


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        self.masked_self_attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout_p)
        self.encoder_decoder_attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout_p)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Linear(dim_model * 4, dim_model),
            nn.Dropout(dropout_p)
        )

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None):
        # Masked Self-Attention
        tgt2, _ = self.masked_self_attention(tgt, tgt, tgt, 
                                             attn_mask=tgt_mask,
                                             key_padding_mask=tgt_pad_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        # print(f"[Decoder] after masked self attention, tgt.shape: {tgt.shape}")

        # Encoder-Decoder Attention
        tgt2, _ = self.encoder_decoder_attention(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        # print(f"[Decoder] after encoder-decoder attention, tgt.shape: {tgt.shape}")

        # Feed Forward
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        # print(f"[Decoder] after feed forward network, tgt.shape: {tgt.shape}")

        return tgt
    

class Encoder(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p, num_layers, device):
        super().__init__()

        self.pos_enc = PositionalEncoding(dim_model, dropout_p, max_len=5000)

        self.layers = nn.ModuleList([
            EncoderLayer(dim_model, num_heads, dropout_p) for _ in range(num_layers)
        ])

        self.device = device
    
    def forward(self, src, src_mask=None):
        src = self.pos_enc(src)             # (bs, num_frames+num_objs+1, dim_model)
        src = torch.permute(src, (1,0,2))
        # print("## Encoder: src", src.shape)

        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src


class Decoder(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p, num_layers, device):
        super().__init__()
    
        self.pos_enc = PositionalEncoding(dim_model, dropout_p, max_len=5000)

        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, num_heads, dropout_p) for _ in range(num_layers)
        ])

        self.device = device

    def forward(self, tgt, enc_src, tgt_mask=None, tgt_pad_mask=None):
        tgt = self.pos_enc(tgt)
        tgt = torch.permute(tgt, (1,0,2)) 
        # print("## Decoder: tgt", tgt.shape)

        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask, tgt_pad_mask)

        return tgt


class MotionTransformer(nn.Module):
    def __init__(
            self, 
            num_tokens, 
            dim_pose=72,
            dim_object=3,
            dim_description=384,
            dim_motion=76,
            dim_model=512, 
            num_heads=8, 
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dropout_p=0.5, 
            device=None,
            stop_signal=None
        ):
        super().__init__()
        # embed each input
        # self.initial_pose_embedding = nn.Linear(in_features=dim_pose, out_features=dim_model)
        # self.object_embedding = nn.Linear(in_features=dim_object, out_features=dim_model)
        # self.motion_description_embedding = nn.Linear(in_features=dim_description, out_features=dim_model)
        # self.next_motion_embedding = nn.Linear(in_features=dim_motion, out_features=dim_model)
        self.num_heads = num_heads

        self.initial_pose_embedding = nn.Linear(in_features=dim_pose, out_features=dim_model)
        self.object_embedding = nn.Linear(in_features=dim_object, out_features=dim_model)
        self.description_embedding = nn.Linear(in_features=dim_description, out_features=dim_model)
        self.motion_embedding = nn.Linear(in_features=dim_motion, out_features=dim_model)

        self.encoder = Encoder(dim_model=dim_model,
                               num_heads=num_heads,
                               dropout_p=dropout_p,
                               num_layers=num_encoder_layers,
                               device=device)
        
        self.decoder = Decoder(dim_model=dim_model,
                               num_heads=num_heads,
                               dropout_p=dropout_p,
                               num_layers=num_decoder_layers,
                               device=device)
        
        self.out = nn.Linear(dim_model, num_tokens)
        
        self.device = device

    def forward(self, initial_pose, obj_bboxes, motion_desc, tgt, tgt_mask=None):
        '''
        initial_pose: (bs, 1, 72)
        object_boxes: (bs, num_objs, 3)
        next_motion_desc: (bs, 384)
        gt_motion: (bs, num_frames, 76)
        '''
        max_seq_len = 61
        # print("## MotionTransformer: initial_pose", initial_pose.shape)
        # print("## MotionTransformer: obj_bboxes", obj_bboxes.shape)
        # print("## MotionTransformer: motion_desc", motion_desc.unsqueeze(1).shape)
        
        # Embedding
        initial_pose_ = self.initial_pose_embedding(initial_pose)
        obj_bboxes_ = self.object_embedding(obj_bboxes)
        motion_desc_ = self.description_embedding(motion_desc).unsqueeze(1)
        # gt_motion_ = self.gt_motion_embedding(gt_motion)

        # context
        src = torch.cat([initial_pose_, motion_desc_, obj_bboxes_], dim=1) # (bs, num_frames+num_objs+1, dim_model)
        # print("## src", src.shape)
        
        '''Encoder'''
        # src_key_padding_mask = self.get_pad_mask(gt_motion)
        memory = self.encoder(src)
        
        '''Decoder'''
        tgt = self.motion_embedding(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.out(output)
        output = output.permute(1, 0, 2)
        return output
    
    def pad_motion(self, motion, max_seq_len=61):
        # print(f"Next motion sequence length: {next_motion.size(1)}")
        padding = (motion[:, -1, :].unsqueeze(0)).repeat(1, max_seq_len - motion.size(1), 1) 
        padded_seq = torch.cat([motion, padding], dim=1)
        # print(f"Padded next motion sequence length: {padded_seq.size(1)}")
        # print("next motion:", next_motion.shape)
        # print("padded_seq:", p)
        return padded_seq
    
    # def get_src_mask(self, src):
    #     src_mask = (src != self.)

    def get_tgt_mask(self, tgt):
        size = tgt.size(1)
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask==0, float('-inf')) # convert zeros to -inf
        mask = mask.masked_fill(mask==1, float(0.0))    # convert ones to 0
        # print("tgt mask:", mask.shape)
        # print(mask)
        return mask.to(self.device)
    
    def get_pad_mask(self, padded_seq):
        # Assume padding is done by repeating the last valid token
        # Check if each element in the sequence is equal to the last element in the padded sequence
        pad_token = padded_seq[:, -1, :].unsqueeze(1)  # Shape: (batch_size, 1, dim)
        pad_mask = torch.all(padded_seq == pad_token, dim=-1)  # Shape: (batch_size, seq_len)
        # print("pad mask", pad_mask)
        
        # Get the first occurrence of the pad_token and mark it as False
        pad_mask_shifted = torch.cat([torch.zeros_like(pad_mask[:, :1]), pad_mask[:, :-1]], dim=1)
        start_pad_token = pad_mask & ~pad_mask_shifted
        # print("first pad mask", start_pad_token)

        # Set the first occurrence to False
        pad_mask = pad_mask & ~start_pad_token
        # print('pad mask:', pad_mask)
        # print("fianl pad mask", pad_mask)
        # print()
        return pad_mask.float().to(self.deivce)