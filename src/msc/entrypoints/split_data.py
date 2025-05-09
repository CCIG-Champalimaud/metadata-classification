import argparse
import polars as pl
from numpy.random import default_rng

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits a CSV-type file into two different sets."
    )

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--id_col", default=None, type=str)
    parser.add_argument("--output_paths", nargs=2, type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--random_seed", default=42, type=int)

    args = parser.parse_args()

    extension = args.input_path.split(".")[-1]
    if extension == "csv" or extension == "tsv":
        sep = "," if extension == "csv" else "\t"
        df = pl.read_csv(args.input_path, sep=sep)
    elif extension == "parquet":
        df = pl.read_parquet(args.input_path)

    df.with_columns([pl.col(pl.String).replace('"', "")])

    rng = default_rng(seed=args.random_seed)

    if args.id_col is None:
        split = rng.choice(
            df.shape[0], int(args.split_ratio * df.shape[0]), replace=False
        )
        split_a = split
        split_b = [i for i in range(df.shape[0]) if i not in split_a]
        df_a = df[split_a]
        df_b = df[split_b]
    else:
        unique_ids = df[args.id_col].unique()
        id_split = rng.choice(
            unique_ids, int(args.split_ratio * len(unique_ids)), replace=False
        )
        id_split_a = id_split
        id_split_b = [i for i in unique_ids if i not in id_split]
        df_a = df.filter(pl.col(args.id_col).is_in(id_split_a))
        df_b = df.filter(pl.col(args.id_col).is_in(id_split_b))

    output_extension = args.output_paths[0].split(".")[-1]
    if output_extension == "csv":
        df_a.write_csv(args.output_paths[0])
        df_b.write_csv(args.output_paths[1])
    elif output_extension == "tsv":
        df_a.write_csv(args.output_paths[0], sep="\t")
        df_b.write_csv(args.output_paths[1], sep="\t")
    elif output_extension == "parquet":
        df_a.write_parquet(args.output_paths[0])
        df_b.write_parquet(args.output_paths[1])
