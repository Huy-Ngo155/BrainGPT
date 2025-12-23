import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, get_rank, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=32768, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, seq_len=None):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.use_flash = False
        try:
            from flash_attn import flash_attn_qkvpacked_func
            self.flash_attn = flash_attn_qkvpacked_func
            self.use_flash = True
        except ImportError:
            pass
    
    def forward(self, x, attention_mask=None, kv_cache=None):
        B, T, C = x.shape
        
        qkv = self.wqkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        kv_len = k.shape[2]
        cos, sin = self.rotary_emb(kv_len)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.use_flash and attention_mask is None and kv_cache is None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            qkv = torch.stack([q, k, v], dim=2)
            attn_output = self.flash_attn(
                qkv,
                dropout_p=self.drop.p if self.training else 0.0,
                causal=True
            )
            attn_output = attn_output.transpose(1, 2)
        else:
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, None, None, :]
                attn_weights = attn_weights + attention_mask
            
            causal_mask = torch.ones(T, kv_len, device=x.device).triu(diagonal=kv_len-T+1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.drop(attn_weights)
            attn_output = attn_weights @ v
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)
        
        return output, (k, v)

class BrainBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, attention_mask=None, kv_cache=None):
        residual = x
        x = self.ln1(x)
        attn_out, new_kv_cache = self.attn(x, attention_mask, kv_cache)
        x = residual + attn_out
        
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x, new_kv_cache

class BrainGPT(nn.Module):
    def __init__(self, vocab_size, dim=768, n_layers=12, n_heads=12, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            BrainBlock(dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, kv_caches=None):
        B, T = input_ids.shape
        
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds maximum {self.max_seq_len}")
        
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        
        new_kv_caches = []
        
        for block, kv_cache in zip(self.blocks, kv_caches):
            x, new_kv_cache = block(x, attention_mask, kv_cache)
            new_kv_caches.append(new_kv_cache)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, new_kv_caches

class BrainTrainer:
    def __init__(self, model, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, grad_clip=1.0):
        self.model = model
        self.grad_clip = grad_clip
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.step_count = 0
        
    def train_step(self, batch, grad_accum=1):
        input_ids, targets = batch
        
        with torch.cuda.amp.autocast():
            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) / grad_accum
        
        self.scaler.scale(loss).backward()
        
        self.step_count += 1
        
        if self.step_count % grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return loss.item() * grad_accum

def setup_ddp():
    if torch.cuda.device_count() > 1:
        init_process_group(backend='nccl')
        rank = get_rank()
        world_size = get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    return rank, world_size

class MemoryProfiler:
    def __init__(self):
        self.reset()
    
    def reset(self):
        torch.cuda.reset_peak_memory_stats()
    
    def measure(self, func, *args, **kwargs):
        self.reset()
        result = func(*args, **kwargs)
        memory = torch.cuda.max_memory_allocated() / 1024**3
        return result, memory

def export_model(model, sample_input, path):
    model.eval()
    
    class ExportWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            logits, _ = self.model(input_ids)
            return logits
    
    wrapper = ExportWrapper(model)
    
    torch.onnx.export(
        wrapper,
        sample_input,
        path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'logits': {0: 'batch', 1: 'seq_len'}
        },
        opset_version=14,
        do_constant_folding=True
    )

def main():
    rank, world_size = setup_ddp()
    
    torch.manual_seed(42 + rank)
    
    model = BrainGPT(vocab_size=50000).cuda()
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    trainer = BrainTrainer(model)
    
    print(f"[Rank {rank}] Model initialized")
    
    if rank == 0:
        profiler = MemoryProfiler()
        sample = torch.randint(0, 50000, (2, 512)).cuda()
        
        with torch.no_grad():
            _, memory = profiler.measure(model, sample)
        
        print(f"Memory usage: {memory:.2f}GB")
        
        export_model(model, sample[:1, :128], "brain_model.onnx")
        print("Model exported to brain_model.onnx")

if __name__ == "__main__":
    main()
