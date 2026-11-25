import math
import torch
import pytest
import nanochat.dataloader as dataloader
from nanochat.loss_eval import evaluate_bpb


class Tokenizer:
    def get_special_tokens(self):
        return {'<|bos|>', '<|eos|>'}

    def get_bos_token_id(self):
        return 0

    def encode_special(self, name):
        assert name == '<|eos|>'
        return 1

    def encode(self, docs, prepend=None, append=None, num_threads=1):
        return [[prepend] + [ord(c) for c in doc] + [append] for doc in docs]


@pytest.mark.parametrize('batch_size', [1, 3])
def test_complete_documents_cross_rows_without_losing_eos(monkeypatch, batch_size):
    def batches(*_args):
        epoch = 0
        while True:
            epoch += 1
            yield ['abcdef', 'gh'], (0, 0, epoch), 0
    monkeypatch.setattr(dataloader, '_document_batches', batches)
    loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        Tokenizer(), batch_size, 4, 'train', buffer_size=2, device='cpu',
        track_data_exposure=True, document_eos=True)
    outputs = []
    for _ in range(6):
        x, y, state = next(loader)
        outputs.extend(y.reshape(-1).tolist())
    cycle = [0] + list(map(ord, 'abcdef')) + [1, 0] + list(map(ord, 'gh')) + [1]
    expected = (cycle * (1 + len(outputs)//len(cycle) + 1))[1:1+len(outputs)]
    assert outputs == expected
    assert state['unique_text_tokens'] == 8
    assert state['text_tokens'] > state['unique_text_tokens']


def test_bpb_ignores_eos_and_bos_losses():
    class Model:
        def get_device(self): return torch.device('cpu')
        def __call__(self, x, y, loss_reduction):
            assert loss_reduction == 'none'
            return torch.where(y < 2, 1000., 2.)
    x = torch.tensor([[0, 2, 1]])
    y = torch.tensor([[2, 1, 0]])
    assert math.isclose(evaluate_bpb(Model(), [(x, y)], 1, torch.tensor([0, 0, 1])), 2/math.log(2), rel_tol=1e-6)


def test_eos_token_is_opt_in_and_preserves_existing_specials():
    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
    plain = RustBPETokenizer.train_from_iterator(iter(['alpha beta gamma']), 265)
    eos = RustBPETokenizer.train_from_iterator(iter(['alpha beta gamma']), 266, include_eos=True)
    assert plain.get_vocab_size() == 265
    assert plain.get_special_tokens() == set(SPECIAL_TOKENS)
    assert eos.get_vocab_size() == 266
    assert eos.get_special_tokens() == set(SPECIAL_TOKENS) | {'<|eos|>'}
