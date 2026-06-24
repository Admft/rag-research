"""Download and cache BEIR datasets under data/beir/."""

from . import DATASETS, dataset_data_path, normalize_dataset_name
from .beir_lib import load_beir_util, load_generic_data_loader, require_beir_library


def dataset_has_files(path):
    if not path.is_dir():
        return False
    markers = ("corpus.jsonl", "queries.jsonl", "qrels")
    return any((path / marker).exists() for marker in markers)


def download_dataset(name, force=False):
    require_beir_library()
    util = load_beir_util()
    GenericDataLoader = load_generic_data_loader()

    key = normalize_dataset_name(name)
    beir_name = DATASETS[key]["beir_name"]
    data_path = dataset_data_path(key)

    if dataset_has_files(data_path) and not force:
        corpus, _, _ = GenericDataLoader(str(data_path)).load(split="test")
        print(f"Downloading {beir_name}... skipped (already at {data_path}, {len(corpus)} docs)")
        return data_path

    if force and data_path.exists():
        import shutil

        shutil.rmtree(data_path)

    print(f"Downloading {beir_name}...", end=" ", flush=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_name}.zip"
    util.download_and_unzip(url, str(data_path.parent))

    if not dataset_has_files(data_path):
        raise RuntimeError(f"BEIR download failed for {beir_name}: {data_path} is missing corpus files")

    corpus, _, _ = GenericDataLoader(str(data_path)).load(split="test")
    print(f"done ({len(corpus)} docs)")
    return data_path


def download_all(force=False):
    paths = []
    for key in DATASETS:
        paths.append(download_dataset(key, force=force))
    return paths
