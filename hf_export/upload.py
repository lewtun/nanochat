"""Upload a converted folder to an existing private Hub model repository."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    p.add_argument('repo_id')
    p.add_argument('--path-in-repo', default='', help='Destination subfolder within the repository')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    folder = Path(args.folder)
    for name in ('config.json', 'tokenizer.json', 'configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
        if not (folder / name).is_file():
            raise ValueError(f'Missing exported file: {name}')
    if not (folder / 'model.safetensors').is_file() and not (folder / 'model.safetensors.index.json').is_file():
        raise ValueError('Missing safetensors model weights')
    subfolder = args.path_in_repo.strip('/')
    if '..' in Path(subfolder).parts:
        raise ValueError('Invalid destination subfolder')
    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type='model')
    if not info.private:
        raise ValueError('Refusing to upload to a public repository')
    target = f'{subfolder}/config.json' if subfolder else 'config.json'
    if api.file_exists(args.repo_id, target, repo_type='model') and not args.overwrite:
        raise FileExistsError(f'{target} already exists; pass --overwrite to replace it')
    if subfolder:
        # Transformers resolves auto_map modules at the repository root even with subfolder=.
        for name in ('configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
            if api.file_exists(args.repo_id, name, repo_type='model'):
                existing = Path(hf_hub_download(args.repo_id, name, repo_type='model')).read_bytes()
                if existing != (folder / name).read_bytes():
                    raise ValueError(f'Existing root {name} differs; use a separate repository or branch')
            else:
                api.upload_file(path_or_fileobj=str(folder / name), path_in_repo=name, repo_id=args.repo_id, repo_type='model')
    api.upload_folder(repo_id=args.repo_id, repo_type='model', folder_path=str(folder), path_in_repo=subfolder,
                      ignore_patterns=['*.pt', '*.pkl'], delete_patterns=['*.safetensors', '*.safetensors.index.json', 'assets/**', '*.pt', '*.pkl', 'tokenizer*', 'special_tokens_map.json', 'added_tokens.json', 'generation_config.json'] if args.overwrite else None)


if __name__ == '__main__':
    main()
