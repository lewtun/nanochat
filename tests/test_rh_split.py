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
    prepare(source, out, 42, 3)
    assert (out / 'data_manifest.json').read_text() == original
