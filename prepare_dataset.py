import os
import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description="prepare dataset for pytorch")
    parser.add_argument('-d', '--directory', type=str, help='input images directory',
                        default=f"{os.getcwd()}/datasets/Animals-10")
    parser.add_argument('-o', '--output', type=str, help='output directory',
                        default=f"{os.getcwd()}/datasets/animals10")
    parser.add_argument('-s', '--split', type=str,
                        help='train/test/valid split, syntax <train>/<test>/<validation> in percentage. i.e. 60/20/20',
                        default="70/15/15")

    args = parser.parse_args()

    train_f = open(f"{args.output}_train", 'w', encoding='UTF8', newline='')
    train_writer = csv.writer(train_f)
    test_f = open(f"{args.output}_test", 'w', encoding='UTF8', newline='')
    test_writer = csv.writer(test_f)
    validation_f = open(f"{args.output}_validation", 'w', encoding='UTF8', newline='')
    validation_writer = csv.writer(validation_f)

    s = args.split.split('/')
    splits = {
        "train": int(s[0]),
        "test": int(s[1]),
        "validation": int(s[2])
    }
    for idx, c in enumerate(os.listdir(args.directory)):
        images = os.listdir(f"{args.directory}\{c}")
        class_count = len(images)
        for imidx, filename in enumerate(images):
            row = [f"{c}/{filename}", idx + 1]
            if imidx < class_count * splits["train"] / 100:
                train_writer.writerow(row)
            elif imidx < class_count * (splits["train"] + splits["test"]) / 100:
                test_writer.writerow(row)
            else:
                validation_writer.writerow(row)
    train_f.close()
    test_f.close()
    validation_f.close()


if __name__ == '__main__':
    main()
