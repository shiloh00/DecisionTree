#!/usr/bin/python

import csv
import argparse
import os


class DTree:
    """stand for a trained decision tree"""
    root = []

    def predict(self, instance):
        return True

    def load_model(self, model_path):
        print("Loading model from "+model_path+"...")

    def save_model(self, model_path):
        print("Saving model to "+model_path+"...")

    def print_model(self):
        print("print decision tree")


class Dataset:
    """stand for a loaded dataset"""

    data_file = None
    header = None
    data = None

    def __init__(self, path):
        pass
        
    def load_dataset(self):
        print("Loading dataset...")
        pass

    def save_dataset(self, outfile):
        print("Saving dataset to "+outfile+"...")
        pass

    def train_model(self, prune):
        return DTree()

    def predict(self):
        pass

    def validate(self):
        pass


def main(args):
    print(args)
    if args["action"] == "train":
        print("Do training")
        if os.path.isfile(args["input"]):
            print(args["input"]+" is a file")
            dataset = Dataset(args["input"])
            model = dataset.train_model(args["prune"])
            if args["print"]:
                mode.print_model()
            if len(args["output"]) > 0 :
                model.save_model(args["output"])
        else:
            print("wrong input file: "+args["input"])
    elif args["action"] == "validate":
        if os.path.isfile(args["input"]) and os.path.isfile(args["model"]):
            print("Do validation")
        else:
            print("wrong input file: "+args["input"]+"or wrong model file: "+args["model"])
        pass
    elif args["action"] == "predict":
        if os.path.isfile(args["input"]) and os.path.isfile(args["model"]):
            print("Do prediction")
        else:
            print("wrong input file: "+args["input"]+"or wrong model file: "+args["model"])
        pass
    else:
        print("UNKNOWN ACTION: " + args["action"])
    print("Done.")


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser(description="Train, test or validate the input dataset using C4.5 decision tree algorithm")
    opt_parser.add_argument('-a', '--action', dest='action', type=str, default='train', choices=['train','validate','predict'], required=True, help='specify the action')
    opt_parser.add_argument('-i', '--input', dest='input', type=str, default='', help='specify the input file(dataset)')
    opt_parser.add_argument('-o', '--output', dest='output', type=str, default='', help='specify the output file(dataset)')
    opt_parser.add_argument('-m', '--model', dest='model', type=str, default='', help='specify the trained model')
    opt_parser.add_argument('--prune', dest='prune', action='store_true', help='whether or not to prune the tree')
    opt_parser.add_argument('--print', dest='print', action='store_true', help='whether or not to print the generated decision tree')
    args = opt_parser.parse_args()
    params = vars(args)
    main(params)

