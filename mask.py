
import torch

#填充掩码
def create_padding_mask(seq,pad_token_id=0):
    mask=(seq!=pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask

#未来掩码
def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size)).type(torch.bool) 
    return mask  


#组合掩码
def create_decoder_mask(tgt_seq, pad_token_id=0):
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)
    look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1)).to(tgt_seq.device)  

    combined_mask = look_ahead_mask.unsqueeze(0) & padding_mask  
    return combined_mask