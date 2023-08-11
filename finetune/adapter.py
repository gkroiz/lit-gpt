import os
import psutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy, XLAFSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.adapter import GPT, Config, mark_only_adapter_as_trainable, Block, adapter_filter
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, step_csv_logger, chunked_cross_entropy
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor, measure_flops, estimate_flops
from scripts.prepare_alpaca import generate_prompt

eval_interval = 600
save_interval = 1000
eval_iters = 100
log_interval = 1
devices = 4
# change this value to force a maximum sequence length
override_max_seq_length = None

# Hyperparameters
learning_rate = 3e-3
batch_size = 1#64 / devices
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_size = 50000  # train dataset size
num_epochs = 5
max_iters = 100#num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02
warmup_steps = 2 * (epoch_size // micro_batch_size) // devices // gradient_accumulation_iters  # 2 epochs

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/adapter/alpaca"),
    precision: Optional[str] = None,
    tpu: bool = False,
    use_seq_shard: bool = False,
    shard_loop: bool = False,
):
    if precision is None:
        precision = "32-true" if tpu else "bf16-mixed"
    fabric_devices = devices
    if fabric_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            fabric_devices = "auto"
            strategy = XLAFSDPStrategy(compute_dtype = torch.bfloat16)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, use_seq_shard, shard_loop)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path, use_seq_shard: bool, shard_loop: bool):
    assert not (shard_loop and use_seq_shard)
    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        # Required for `pjrt://` init_method
        import torch_xla.experimental.pjrt_backend
        import torch.distributed as dist
        import torch_xla.debug.profiler as xp
        
        server = xp.start_server(3294)

        dist.init_process_group('xla', init_method='pjrt://')
        group_gloo = dist.new_group(ranks = [_ for _ in range(fabric.world_size)], backend="gloo")

    fabric.print(hparams)
    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    config = Config.from_name(name=checkpoint_dir.name, adapter_start_layer=0)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    if use_seq_shard:
        fabric.print(f'Memory usage before model init sequential: {(psutil.virtual_memory()[3]/1e9):.02f} GB')
        model = sequential_shard(fabric, config, checkpoint_path, group_gloo)
        fabric.print(f'Memory usage before model init (pre-shard): {(psutil.virtual_memory()[3]/1e9):.02f} GB')
  
    elif shard_loop:
        fabric.print(f'Memory usage before model init (pre-shard): {(psutil.virtual_memory()[3]/1e9):.02f} GB')
        model = sequential_shard(fabric, config, checkpoint_path)
        fabric.print(f'Memory usage after model init (post-shard): {(psutil.virtual_memory()[3]/1e9):.02f} GB')

    else:
        fabric.print(f'Memory usage before model init (pre-shard): {(psutil.virtual_memory()[3]/1e9):.02f} GB')
        model = regular_shard(fabric, config, checkpoint_path)
        fabric.print(f'Memory usage after model init (post-shard): {(psutil.virtual_memory()[3]/1e9):.02f} GB')
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    fabric.print(f"Number of trainable parameters: {num_params:,}")
    num_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    fabric.print(f"Number of non trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    if fabric.device.type == "xla":
        optimizer = fabric.setup_optimizers(optimizer)
    else:
        model, optimizer = fabric.setup(model, optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.time()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir, speed_monitor)
    fabric.print(f"Training time: {(time.time()-train_time):.2f}s")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_adapter_finetuned.pth"
    save_adapter_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)

    # validate(fabric, model, val_data, tokenizer, longest_seq_length)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # estimated is too much of an optimistic estimate, left just for reference
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.time()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    for iter_num in range(max_iters):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids, targets = get_batch(
            fabric, train_data, longest_seq_length, longest_seq_ix if iter_num == 0 else None
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, max_seq_length=max_seq_length, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        t1 = time.time()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(fabric, model, val_data, tokenizer, longest_seq_length)
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_adapter_checkpoint(fabric, model, checkpoint_path)


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, longest_seq_length)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=model.device)
    max_returned_tokens = len(encoded) + 100
    output = generate(
        model, idx=encoded, max_returned_tokens=max_returned_tokens, max_seq_length=max_returned_tokens, temperature=0.8
    )
    output = tokenizer.decode(output)
    fabric.print(output)

    model.reset_cache()

    model.train()
    return val_loss.item()


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_length: int, longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # it's better to pad to a fixed seq length with XLA to avoid recompilation
    max_len = max(len(s) for s in input_ids) if fabric.device.type != "xla" else longest_seq_length

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_adapter_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving adapter weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": adapter_filter})


def sequential_shard(fabric, config, checkpoint_path, group_gloo):
    if fabric.device.type == "xla":
        from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel
        from torch_xla.distributed.fsdp import checkpoint_module
        import torch.distributed as dist

    with torch.device("meta"):
        model = GPT(config)
    if fabric.global_rank == 0:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    fabric.barrier('wait')
    fabric.print(f'Memory usage after loading checkpoint onto master rank: {(psutil.virtual_memory()[3]/1e9):.02f} GB')

    # replace lm_head
    fabric.print('replace lm_head')
    with torch.device("cpu"):
        lm_head_on_device = nn.Linear(model.config.n_embd, config.padded_vocab_size, bias=False)
    for param_name, param in model.lm_head.named_parameters():
        key = f"lm_head.{param_name}"
        broadcast_param = state_dict[key] if fabric.global_rank == 0 else torch.empty_like(param, dtype=torch.bfloat16, device="cpu")
        broadcast_param = broadcast_param.type(torch.float32)
        dist.broadcast(tensor=broadcast_param, src=0, group=group_gloo)
        broadcast_param = broadcast_param.type(torch.bfloat16)
        keys = lm_head_on_device.load_state_dict({param_name: broadcast_param}, strict=False)
        assert not keys.unexpected_keys
    model.lm_head = lm_head_on_device

    # replace wte
    fabric.print('replace transformer.wte')
    with torch.device("cpu"):
        wte_on_device = nn.Embedding(model.config.padded_vocab_size, model.config.n_embd)
    for param_name, param in model.transformer.wte.named_parameters():
        key = f"transformer.wte.{param_name}"
        broadcast_param = state_dict[key] if fabric.global_rank == 0 else torch.empty_like(param, dtype=torch.bfloat16, device="cpu")
        broadcast_param = broadcast_param.type(torch.float32)
        dist.broadcast(tensor=broadcast_param, src=0, group=group_gloo)
        broadcast_param = broadcast_param.type(torch.bfloat16)
        keys = wte_on_device.load_state_dict({param_name: broadcast_param}, strict=False)
        assert not keys.unexpected_keys
    model.transformer.wte = wte_on_device

    # replace ln_f
    fabric.print('replace transformer.ln_f')
    with torch.device("cpu"):
        ln_f_on_device = model.config.norm_class(model.config.n_embd, eps=model.config.norm_eps)
    key = 'transformer.ln_f.weight'
    for param_name, param in model.transformer.ln_f.named_parameters():
        key = f"transformer.ln_f.{param_name}"
        broadcast_param = state_dict[key] if fabric.global_rank == 0 else torch.empty_like(param, dtype=torch.bfloat16, device="cpu")
        broadcast_param = broadcast_param.type(torch.float32)
        dist.broadcast(tensor=broadcast_param, src=0, group=group_gloo)
        broadcast_param = broadcast_param.type(torch.bfloat16)
        keys = ln_f_on_device.load_state_dict({param_name: broadcast_param}, strict=False)
        assert not keys.unexpected_keys
    model.transformer.ln_f = ln_f_on_device

    # replace all blocks
    for i in range(config.n_layer):
        fabric.print(f'replace transformer.h[{i}]')
        with torch.device("cpu"):
            block_on_device = Block(model.config, i)
        for param_name, param in block_on_device.named_parameters():
            key = f"transformer.h.{i}.{param_name}"
            if adapter_filter(key, None):
                broadcast_param = param
            else:
                broadcast_param = state_dict[key] if fabric.global_rank == 0 else torch.empty_like(param, dtype=torch.bfloat16, device="cpu")
            broadcast_param = broadcast_param.type(torch.float32)
            dist.broadcast(tensor=broadcast_param, src=0, group=group_gloo)
            broadcast_param = broadcast_param.type(torch.bfloat16)
            keys = block_on_device.load_state_dict({param_name: broadcast_param}, strict=False)
            assert not keys.unexpected_keys
        model.transformer.h[i] = XlaFullyShardedDataParallel(checkpoint_module(block_on_device), disable_reshard_on_root=False, compute_dtype=torch.bfloat16)

    mark_only_adapter_as_trainable(model)
    model = fabric.setup(model)
    return model


def shard_loop(fabric, config, checkpoint_path):
    for local_rank in range(devices):
        if fabric.local_rank == local_rank:
            print(f'loading on local rank {local_rank}')
            model = regular_shard(fabric, config, checkpoint_path)
        fabric.barrier('wait')
    return model


def regular_shard(fabric, config, checkpoint_path):
    with torch.device("cpu"):
        model = GPT(config)
        model.apply(model._init_weights)  # for the adapter weights
                
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to adapter weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    mark_only_adapter_as_trainable(model)

    model = fabric.setup_module(model)      


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
