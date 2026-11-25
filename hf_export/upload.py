"""Upload a converted folder to an existing private Hub model repository."""
import argparse
import glob
import posixpath
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder'); p.add_argument('repo_id')
    p.add_argument('--path-in-repo', default='', help='Destination subfolder within the repository')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    folder = Path(args.folder)
    for name in ('config.json', 'tokenizer.json', 'configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
        if not (folder / name).is_file():
            raise ValueError(f'Missing exported file: {name}')
    if not (folder / 'model.safetensors').is_file() and not (folder / 'model.safetensors.index.json').is_file():
        raise ValueError('Missing safetensors model weights')
    if args.path_in_repo.startswith('/') or '\\' in args.path_in_repo or '..' in args.path_in_repo.split('/'):
        raise ValueError('Invalid destination subfolder')
    subfolder = posixpath.normpath(args.path_in_repo)
    if subfolder == '.':
        subfolder = ''
    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type='model')
    if not info.private:
        raise ValueError('Refusing to upload to a public repository')
    target = f'{subfolder}/config.json' if subfolder else 'config.json'
    if api.file_exists(args.repo_id, target, repo_type='model', revision=info.sha) and not args.overwrite:
        raise FileExistsError(f'{target} already exists; pass --overwrite to replace it')
    remote = api.list_repo_files(args.repo_id, repo_type='model', revision=info.sha)
    has_runs = any(p.endswith('/config.json') for p in remote)
    if subfolder or has_runs:
        # All subfolder models share these root files in Transformers 5.24.
        for name in ('configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
            if api.file_exists(args.repo_id, name, repo_type='model', revision=info.sha):
                existing = Path(hf_hub_download(args.repo_id, name, repo_type='model', revision=info.sha)).read_bytes()
                if existing != (folder / name).read_bytes():
                    raise ValueError(f'Existing root {name} differs; use a separate repository or branch')
            elif has_runs:
                raise ValueError(f'Cannot verify shared root code: {name} is missing')
            elif subfolder:
                api.upload_file(path_or_fileobj=str(folder / name), path_in_repo=name, repo_id=args.repo_id, repo_type='model')
    # Enumerate exact stale paths in this destination, never glob across run directories.
    stale = []
    prefix = subfolder + '/' if subfolder else ''
    if args.overwrite:
        for remote_path in remote:
            if not remote_path.startswith(prefix):
                continue
            relative = remote_path[len(prefix):]
            path = Path(relative)
            top_level = len(path.parts) == 1
            standard = (path.name.startswith('model') and (path.name.endswith('.safetensors') or path.name.endswith('.safetensors.index.json'))) or path.name.startswith('tokenizer') or path.name in ('special_tokens_map.json', 'added_tokens.json', 'generation_config.json') or path.suffix in ('.pt', '.pkl')
            if (top_level and standard) or relative.startswith('assets/'):
                stale.append(glob.escape(relative))
    # The released card is hand-edited; later weight exports must not replace it.
    ignore = ['*.pt', '*.pkl']
    if prefix + 'README.md' in remote:
        ignore.append('README.md')
    api.upload_folder(repo_id=args.repo_id, repo_type='model', folder_path=str(folder), path_in_repo=subfolder,
                      ignore_patterns=ignore, delete_patterns=stale or None)


if __name__ == '__main__':
    main()
