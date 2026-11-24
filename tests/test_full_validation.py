import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.loss_eval import full_document_validation_batches


class FakeTokenizer:
    def get_bos_token_id(self):
        return 1

    def encode_special(self, token):
        assert token == '<|eos|>'
        return 2

    def encode(self, texts):
        return [[int(word) for word in text.split()] for text in texts]


@pytest.mark.parametrize('world_size', [1, 2, 3, 8])
def test_full_validation_scores_each_text_token_once(tmp_path, world_size):
    docs = ['3 4 5 6 7 8 9', '10', '11 12 13', '14 15', '', '16 17 18 19']
    path = tmp_path / 'val.parquet'
    pq.write_table(pa.table({'text': docs}), path, row_group_size=2)
    scored = []
    for rank in range(world_size):
        batches = full_document_validation_batches(FakeTokenizer(), path, 2, 3, rank, world_size, 'cpu')
        for x, y in batches:
            assert x.shape == y.shape and x.shape[1] >= 2
            scored.extend(i for i in y.flatten().tolist() if i >= 3)
    assert sorted(scored) == list(range(3, 20))
