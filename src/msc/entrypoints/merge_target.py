from ..data_loading import read_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merges a feature dataset with a target dataset"
    )

    parser.add_argument("--input_path", required=True, type=str, nargs="+")
    parser.add_argument("--class_path", required=True, type=str, nargs="+")
    parser.add_argument("--merge_cols", required=True, type=str, nargs="+")
    parser.add_argument("--target_col", default="class", type=str)
    parser.add_argument("--output_path", required=True, type=str)

    args = parser.parse_args()

    df = read_data(args.input_path, group_cols=None)
    classification_data = read_data(args.class_path, group_cols=None)
    classification_data = classification_data.select(
        [*args.merge_cols, args.target_col]
    )
    df = df.join(classification_data, on=args.merge_cols, how="inner")

    output_extension = args.output_path.split(".")[-1]
    if output_extension == "csv":
        df.write_csv(args.output_path)
    elif output_extension == "tsv":
        df.write_csv(args.output_path, sep="\t")
    elif output_extension == "parquet":
        df.write_parquet(args.output_path)
