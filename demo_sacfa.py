import xformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# Some code are omitted for brevity
class CrossAttention(nn.Module):
    """
    Implements a cross-attention mechanism with special handling for spatial key-value pairs and
    subject-aware cross-frame attention (SACFA).
    
    Args:
        upblock: Flag indicating if the current block is an upsample block.
        attn_map: An object to store cross-attention maps in down and mid blocks.
    """

    def __init__(self, upblock=False, attn_map=None):
        super().__init__()

        # For cross-attention masks
        self.height = 320
        self.width = 512
        self.factor_b = 40 * 60

        # SACFA parameters
        self.attn_map = attn_map  # Store cross-attn maps in down and mid blocks
        self.factors = [2, 4]  # Specific down_sample factors for which key-value pairs are stored
        self.upblock = upblock

    def default(self, val, d):
        """
        Utility function to return the value if it is not None, otherwise return the default.
        """
        return val if val is not None else d
        
    def efficient_forward_all_frames(self, x, context=None, mask=None, window=None, sacfa_windows=None, guide_windows=None, is_cfg=False):
        """
        Forward pass for all frames with efficient memory usage, handling special cases where 
        the current window is in the self-recurrent window.

        Args:
            x: Input tensor.
            context: Context tensor for cross-attention.
            mask: Mask tensor to apply for attention.
            window: Current window index.
            sacfa_windows: Windows where SACFA should be applied.
            guide_windows: Windows where guidance is applied.
            is_cfg: Whether it is a classifier-free guidance step.
        """
        # If the current window is not in the self-recurrent window
        if window not in guide_windows:
            out = self.efficient_forward(x, context, mask=None)  # Normal forward pass
            return out

        # If the current window is in the self-recurrent window
        b, n, d = x.shape
        seq_len = context.shape[1] if context is not None else n
        is_cross = True if context is not None else False  # Determine if it is a cross-attention

        context = self.default(context, x)
        
        # Compute queries, keys, and values
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        head_to_batch_dim = lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads).contiguous()
        down_sample_factor = int(int(self.factor_b // n)**0.5)

        # Store key-value pairs for specific layers in the upsample block
        if not is_cross and self.upblock and not is_cfg and down_sample_factor in self.factors and self.store_kv:
            self.attn_map[window].set_cur_key(k, down_sample_factor)  # Temporarily store the key
            self.attn_map[window].set_clean_reference_k(k, down_sample_factor)  # Store clean reference

        # Store attention maps for cross-attention layers in specific conditions
        if is_cross and not self.upblock and not is_cfg and down_sample_factor in self.factors:
            q, k, v = map(head_to_batch_dim, (q, k, v))
            attention_score = self.get_attn_score(q, k, mask)  # Compute attention scores
            out = torch.bmm(attention_score, v)  # Apply attention
            out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads).contiguous()
            
            self.attn_map[window].capture_attention_map(attention_score, b, self.heads, q, k, v, self.factors)  # Store attn map
            out = self.to_out(out)
            return out
        
        else:
            is_sacfa = False
            # Check if the current layer has a Subject mask input
            if isinstance(mask, tuple):
                mask, sacfa_masks = mask
                is_sacfa = True 
            
            # If SACFA is enabled for the current window
            if is_sacfa and window in sacfa_windows:
                # Concatenate key-value tokens from all frames
                all_k = k.reshape(b * seq_len, -1)
                all_v = v.reshape(b * seq_len, -1)
                
                mask_size = (int(40 // down_sample_factor), int(64 // down_sample_factor)   )
                sacfa_masks = sacfa_masks.unsqueeze(1)
                
                # Resize the SACFA mask to the required size
                if sacfa_masks.shape[-2:] != mask_size:
                    sacfa_masks = F.interpolate(sacfa_masks, size=mask_size, mode='bilinear', align_corners=False)
                
                sacfa_masks = sacfa_masks.view(b * seq_len)
                
                # Extract tokens related to the subject
                # These corresponds to K' and V' in the paper for SACFA computation(Eq. 4)
                k_tokens = all_k[sacfa_masks.bool()].unsqueeze(0).repeat(b, 1, 1)
                v_tokens = all_v[sacfa_masks.bool()].unsqueeze(0).repeat(b, 1, 1)
                
                # Concatenate subject tokens with the original key and value
                k = torch.cat([k, k_tokens], dim=1).contiguous()
                v = torch.cat([v, v_tokens], dim=1).contiguous()
            
            # Perform the normal forward pass
            q, k, v = map(head_to_batch_dim, (q, k, v))
            out = xformers.ops.memory_efficient_attention(q, k, v, op=None)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads).contiguous()
                        
            out = self.to_out(out)
            return out
