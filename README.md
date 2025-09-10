```python
#!/usr/bin/env python
import os, torch, random, gc
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
import math

# ------------------------
# Dynamic Sparse Mask Utils
# ------------------------
def apply_mask(model, mask_dict):
    """Zero out pruned connections."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.data *= mask_dict[name]

def get_mask(model, sparsity=0.8, use_gradients=False):
    """Initialize mask using weight or gradient magnitudes."""
    mask_dict = {}
    param_dict = dict(model.named_parameters())
    for name, param in param_dict.items():
        if "lora_" in name and param.requires_grad:
            numel = param.numel()
            k = int(numel * (1 - sparsity))
            if k < 1:
                continue
            if use_gradients and param.grad is not None:
                flat = param.grad.detach().abs().view(-1)
            else:
                flat = param.data.abs().view(-1)
            topk = torch.topk(flat, k).values.min()
            mask = (flat >= topk).float().view_as(param.data)
            mask_dict[name] = mask
    return mask_dict

def update_mask(model, old_mask, drop_fraction=0.1):
    """RigL update: prune random active weights, regrow by gradient magnitude."""
    new_mask = {}
    param_dict = dict(model.named_parameters())
    for name, mask in old_mask.items():
        param = param_dict[name]
        grad = param.grad
        if grad is None:
            new_mask[name] = mask.clone()
            continue

        mask = mask.clone().view(-1)
        grad = grad.detach().view(-1)

        # prune some active connections
        active = (mask == 1).nonzero(as_tuple=True)[0]
        drop_k = int(len(active) * drop_fraction)
        if drop_k > 0:
            drop_idx = active[torch.randperm(len(active))[:drop_k]]
            mask[drop_idx] = 0

        # regrow using largest gradients
        inactive = (mask == 0).nonzero(as_tuple=True)[0]
        if len(inactive) > 0 and drop_k > 0:
            grad_inactive = grad[inactive].abs()
            grow_k = min(drop_k, len(inactive))
            grow_idx = grad_inactive.topk(grow_k).indices
            mask[inactive[grow_idx]] = 1

        new_mask[name] = mask.view_as(old_mask[name])
    return new_mask

# ------------------------
# Evaluation (Perplexity)
# ------------------------
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    losses = []
    for batch in dataloader:
        inputs = {k: v.cuda() for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.mean()
        losses.append(loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses))  # perplexity

# ------------------------
# Main Training Loop
# ------------------------
def main():
    model_name = "gpt2"
    train_ds = load_dataset("imdb", split="train[:1%]")   # demo
    val_ds = load_dataset("imdb", split="test[:1%]")      # small validation

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    base_model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()

    # -------------------
    # Warmup for gradient-based mask init
    # -------------------
    print("🔄 Warmup step for gradient-based mask init...")
    batch = next(iter(train_loader))
    inputs = {k: v.cuda() for k, v in batch.items()}
    with torch.cuda.amp.autocast():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.mean()
    scaler.scale(loss).backward()
    mask_dict = get_mask(model, sparsity=0.8, use_gradients=True)
    apply_mask(model, mask_dict)
    optimizer.zero_grad()
    print("✅ Gradient-based mask initialized")

    # -------------------
    # Training loop
    # -------------------
    steps = 0
    for epoch in range(1):  # demo
        for batch in train_loader:
            inputs = {k: v.cuda() for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.mean()

            scaler.scale(loss).backward()

            # RigL update before optimizer clears grads
            if steps % 100 == 0 and steps > 0:
                mask_dict = update_mask(model, mask_dict, drop_fraction=0.1)

            clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            apply_mask(model, mask_dict)

            steps += 1
            if steps % 10 == 0:
                print(f"Step {steps} | Loss {loss.item():.4f}")

            if steps % 200 == 0:
                ppl = evaluate(model, val_loader)
                print(f"🔎 Validation Perplexity: {ppl:.2f}")
                os.makedirs("./dyn-sparse-peft", exist_ok=True)
                model.save_pretrained("./dyn-sparse-peft")
                tokenizer.save_pretrained("./dyn-sparse-peft")
                print("💾 Saved checkpoint")

            if steps % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

main()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------------
# Load Model + Tokenizer
# ------------------------
checkpoint_dir = "./dyn-sparse-peft"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

# Load LoRA adapter
model = PeftModel.from_pretrained(model, checkpoint_dir).cuda()
model.eval()

# ------------------------
# Test Prompt
# ------------------------
prompt = "The movie was absolutely fantastic because"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

print("=== Prompt ===")
print(prompt)
print("\n=== Generated ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
