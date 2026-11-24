"""Upload an exported model to a private Hub repo (requires prior operator approval)."""
import argparse
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    p.add_argument('repo_id')
    args = p.parse_args()
    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type='model')
    if not info.private:
        raise ValueError('Refusing to upload to a public repository')
    api.upload_folder(repo_id=args.repo_id, repo_type='model', folder_path=args.folder)


if __name__ == '__main__':
    main()
