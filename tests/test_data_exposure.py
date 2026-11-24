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
