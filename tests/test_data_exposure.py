"""Distinct positions must not count repeated prefixes as fresh data."""
import nanochat.dataloader as dataloader


class Tokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, docs, prepend, num_threads):
        return [[0] + list(range(1, len(doc) + 1)) for doc in docs]


def test_repeated_cropped_documents(monkeypatch):
    def batches(*_args):
        epoch = 0
        while True:
            epoch += 1
            yield ['abc', 'defgh'], (0, 0, epoch), 0

    monkeypatch.setattr(dataloader, '_document_batches', batches)
    loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        Tokenizer(), 1, 4, 'train', buffer_size=2, device='cpu', track_data_exposure=True
    )
    for _ in range(20):
        _, _, state = next(loader)
    assert state['unique_text_tokens'] == 7  # one of eight positions is always cropped
    assert state['text_tokens'] == 61  # includes repeat exposures


def test_eos_exposure_tracks_documents_across_files_row_groups_and_ranks(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = []
    for file_index in range(2):
        path = tmp_path / f'shard_{file_index:05d}.parquet'
        documents = [f'{10 + 8*file_index + 2*i} {11 + 8*file_index + 2*i}' for i in range(4)]
        pq.write_table(pa.table({'text': documents}), path, row_group_size=2)
        paths.append(str(path))
    val = tmp_path / 'shard_99999.parquet'
    pq.write_table(pa.table({'text': ['99']}), val)
    paths.append(str(val))
    monkeypatch.setattr(dataloader, 'list_parquet_files', lambda **kwargs: paths)

    class EosTokenizer:
        def get_special_tokens(self): return {'<|eos|>': 1}
        def get_bos_token_id(self): return 0
        def encode_special(self, _): return 1
        def encode(self, docs, prepend, append, num_threads):
            return [[prepend] + [int(i) for i in doc.split()] + [append] for doc in docs]

    for rank in range(2):
        monkeypatch.setattr(dataloader, 'get_dist_info', lambda: (True, rank, rank, 2))
        loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            EosTokenizer(), 1, 3, 'train', tokenizer_batch_size=1,
            device='cpu', track_data_exposure=True, document_eos=True)
        seen, total = set(), 0
        for _ in range(20):
            _, targets, state = next(loader)
            text = [t for t in targets.flatten().tolist() if t >= 10]
            seen.update(text)
            total += len(text)
            assert state['unique_text_tokens'] == len(seen)
            assert state['text_tokens'] == total
        assert len(seen) == 8
