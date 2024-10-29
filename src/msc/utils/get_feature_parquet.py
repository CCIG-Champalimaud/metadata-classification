import argparse
import polars as pl
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

from ..dicom_feature_extraction import extract_all_features


class DICOMFeatureExtraction:
    def __init__(
        self, metadata_features: bool = True, image_features: bool = True
    ):
        self.metadata_features = metadata_features
        self.image_features = image_features

    def __call__(self, p: str) -> dict:
        """Wraps the metadata extraction from a DICOM directory path.

        Args:
            p (str): path to DICOM directory.

        Returns:
            d (dict): metadata dictionary.
        """
        d = extract_all_features(
            p,
            metadata_features=self.metadata_features,
            image_features=self.image_features,
        )
        if len(d["file_path"]) > 0 and d["seg"] == False and d["valid"] == True:
            return d
        else:
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extracts the metadata of a directory containing DICOM series and \
            prints the data in JSON format."
    )

    parser.add_argument("--input_dir", required=True, nargs="+")
    parser.add_argument(
        "--feature_types",
        nargs="+",
        default=["metadata", "image"],
        choices=["metadata", "image"],
    )
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--individual_pattern", default="*")
    parser.add_argument("--n_workers", default=0, type=int)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    files = []
    for folder in args.input_dir:
        files.extend([str(x) for x in Path(folder).rglob(args.pattern)])
    extractor = DICOMFeatureExtraction(
        metadata_features="metadata" in args.feature_types,
        image_features="image" in args.feature_types,
    )
    output = []
    if args.n_workers < 2:
        it = map(extractor, files)
        with tqdm(it, total=len(files), smoothing=1.0) as pbar:
            for f in pbar:
                if f is not None:
                    output.append(f)
    else:
        with Pool(args.n_workers) as p:
            it = p.imap_unordered(extractor, files)
            with tqdm(it, total=len(files), smoothing=1.0) as pbar:
                for f in pbar:
                    if f is not None:
                        output.append(f)

    pl.DataFrame(output).write_parquet(args.output_path)
