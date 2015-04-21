#!/usr/bin/python

import csv
import argparse


class DTree:
    """stand for a trained decision tree"""
    root = []

    def predict(instance):
        return True

    def load_model(model_path):
        print("Loading model from "+model_path+"...")

    def save_model(model_path):
        print("Saving model to "+model_path+"...")


class Dataset:
    """stand for a loaded dataset"""

    data_file = None
    header = None
    data = None

    def __init__(self, path):
        pass
        
    def load_dataset():
        print("Loading dataset...")
        pass

    def save_dataset(outfile):
        print("Saving dataset to "+outfile+"...")
        pass

    def train_model():
        pass

    def predict():
        pass

    def validate():
        pass


def print_usage():
    print("Usage: ")

def load_model(mode_path):
    print("Loading model...")

def save_model(model, model_path):
    print("Saving model to "+ model_path + "...")

def print_model(model):
    print("The generated decision tree:")

def train_model(dataset):
    pass

def main(args):
    print(args)

if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser(description="Train, test or validate the input dataset using C4.5 decision tree algorithm")
    opt_parser.add_argument('-a', '--action', dest='action', type=str, default='train', choices=['train','validate','predict'], required=True, help='specify the action')
    opt_parser.add_argument('-i', '--input', dest='input', type=str, default='', help='specify the input file(dataset)')
    opt_parser.add_argument('-o', '--output', dest='output', type=str, default='', help='specify the output file(dataset)')
    opt_parser.add_argument('-m', '--model', dest='model', type=str, default='', help='specify the trained model')
    opt_parser.add_argument('-p', '--prune', dest='prune', action='store_true', help='whether or not to prune the tree')
    args = opt_parser.parse_args()
    params = vars(args)
    main(params)

