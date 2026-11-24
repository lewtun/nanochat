import json
from scripts.prepare_rh_data import prepare


def test_leaf_split_with_three_file_level_fallbacks_is_rerunnable(tmp_path):
    source = tmp_path / 'src'
    names = [
        'A.lean', 'B.lean',
        'Formalism/A.lean', 'Formalism/B.lean',
        'Formalism/Area/Leaf/a.lean', 'Formalism/Area/Other/b.lean',
        'Development/ResolutionOfSingularities/A.lean',
        'Development/ResolutionOfSingularities/B.lean',
        'Development/ResolutionOfSingularities/Sub/Leaf/c.lean',
        'Other/Area/Leaf/a.lean', 'Other/Area/Other/b.lean',
    ]
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'-- {name}\n')
    out = tmp_path / 'out'
    prepare(source, out, 42, 3)
    original = (out / 'data_manifest.json').read_text()
    manifest = json.loads(original)
    assert set(manifest['file_level_areas']) == {'(root)', 'Formalism', 'Development/ResolutionOfSingularities'}
    assert all(len(files) == 1 for files in manifest['file_level_areas'].values())
    assert not manifest['uncovered_areas']
    assert manifest['document_eos'] is True
    assert manifest['data_prefix'] == 'base_data'
    assert manifest['validation_shard'] == manifest['parquet_files'][-1]
    assert set(manifest['area_evaluation_set'].values()) == {'matched', 'difficulty_only', 'unseen'}
    assert all(v['train_bytes'] == 0 for k, v in manifest['area_split_bytes'].items() if manifest['area_evaluation_set'][k] == 'unseen')
    assert all(v['train_bytes'] > 0 for k, v in manifest['area_split_bytes'].items() if manifest['area_evaluation_set'][k] != 'unseen')
    assert set(manifest['validation_files']).isdisjoint(manifest['difficulty_files'])
    assert set(manifest['validation_files']).isdisjoint(manifest['unseen_files'])
    prepare(source, out, 42, 3)
    assert (out / 'data_manifest.json').read_text() == original


def test_validation_content_is_removed_from_training(tmp_path):
    import pyarrow.parquet as pq
    source = tmp_path / 'src'
    for name, text in [('Shared/Area/A/x.lean', 'identical'), ('Shared/Area/B/y.lean', 'identical'), ('Shared/Area/B/z.lean', 'different' * 100), ('Shared/Area/top.lean', 'other' * 100)]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    output = tmp_path / 'output'
    prepare(source, output, 42, 100)
    manifest = json.loads((output / 'data_manifest.json').read_text())
    assert manifest['excluded_duplicate_train_files'] == ['Shared/Area/B/y.lean']
    train = pq.read_table(output / 'base_data' / manifest['parquet_files'][0])['text'].to_pylist()
    assert sorted(train) == sorted(['different' * 100, 'other' * 100])
    assert manifest['git_sha']
