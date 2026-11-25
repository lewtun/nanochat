"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    import itertools
    selected_batches = batch_iter if steps is None else itertools.islice(batch_iter, steps)
    for x, y in selected_batches:
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


def full_document_validation_batches(tokenizer, parquet_path, batch_size, sequence_len, rank, world_size, device):
    """Score each validation document once, assigning complete documents by index.

    Document segments overlap by one token so each text target is scored exactly
    once. Padding uses ignored targets; sorting each small buffer reduces padding.
    """
    import pyarrow.parquet as pq

    bos = tokenizer.get_bos_token_id()
    eos = tokenizer.encode_special('<|eos|>')
    pending = []
    index = 0

    def emit(segments):
        width = max(2, max(len(segment) - 1 for segment in segments))
        x = torch.full((len(segments), width), bos, dtype=torch.long, device=device)
        y = torch.full_like(x, -1)
        for row, segment in enumerate(segments):
            length = len(segment) - 1
            x[row, :length] = torch.tensor(segment[:-1], dtype=torch.long, device=device)
            y[row, :length] = torch.tensor(segment[1:], dtype=torch.long, device=device)
        return x, y

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=128, columns=['text']):
        docs = batch.column(0).to_pylist()
        selected = [text for offset, text in enumerate(docs) if (index + offset) % world_size == rank]
        index += len(docs)
        for ids in tokenizer.encode(selected) if selected else []:
            doc = [bos] + ids + [eos]
            pending.extend(doc[start:start + sequence_len + 1] for start in range(0, len(doc) - 1, sequence_len))
        if len(pending) >= batch_size * 8:
            pending.sort(key=len)
            while len(pending) >= batch_size:
                yield emit(pending[:batch_size])
                del pending[:batch_size]
    pending.sort(key=len)
    while pending:
        yield emit(pending[:batch_size])
        del pending[:batch_size]
