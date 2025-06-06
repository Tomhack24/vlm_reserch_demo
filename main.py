import torch
from data.raw.load_datasets import load_coco_datasets


def main():
    val, test = load_coco_datasets()
    print(val)
    print(test)
    print(val[0])


if __name__ == "__main__":
    main()
