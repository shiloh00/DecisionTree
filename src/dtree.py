#!/usr/bin/python

import csv
import argparse
import os
import sys
import math
import json

NUMERIC = "numeric"
NOMINAL = "nominal"

CMP_LT = "CMP_LT"
CMP_EQ = "CMP_EQ"
CMP_GE = "CMP_GE"

#class TNode:
#    """stand for the nodes in decision tree"""
#    name = ""
#    data_type = ""
#    ops = ""
#    value = 0
#    mean_value = 0
#    marjor_value = 0
#    leaf = False
#    label = None
#    subtree = None


class DTree:
    """stand for a trained decision tree"""
    root = None

    def load_model(self, model_path):
        print("Loading model from "+model_path+"...")
        with open(model_path) as fp:
            self.root = json.load(fp)

    def save_model(self, model_path):
        print("Saving model to "+model_path+"...")
        with open(model_path, "w") as fp:
            #json.dump(self.root, fp, indent=4)
            json.dump(self.root, fp)

    def print_model(self):
        print("print decision tree")


class Dataset:
    """stand for a loaded dataset"""

    data_file = None
    header = None
    data = None
    header_type = {
            "winpercent": NUMERIC,
            "oppwinpercent": NUMERIC,
            "weather": NOMINAL,
            "temperature": NUMERIC,
            "numinjured": NUMERIC,
            "oppnuminjured": NUMERIC,
            "startingpitcher": NOMINAL,
            "oppstartingpitcher": NOMINAL,
            "dayssincegame": NUMERIC,
            "oppdayssincegame": NUMERIC,
            "homeaway": NOMINAL,
            "rundifferential": NUMERIC,
            "opprundifferential": NUMERIC,
            "winner": NOMINAL
    }

    header_row = ["winpercent", "oppwinpercent", "weather", "temperature", "numinjured", "oppnuminjured", 
            "startingpitcher", "oppstartingpitcher", "dayssincegame", "oppdayssincegame", "homeaway", 
            "rundifferential", "opprundifferential", "winner"]
    header_index = {}

    target = "winner"
    target_values = []

    def __init__(self, path):
        with open(path, 'rb') as csvfile:
            rd = csv.reader(csvfile, delimiter=',')
            __header_row = next(rd, None)
            if __header_row:
                self.header_row = __header_row
            for idx in range(0, len(self.header_row)):
                self.header_row[idx] = self.header_row[idx].strip()
                self.header_index[self.header_row[idx]] = idx
            print(self.header_row)
            self.data = []
            print("Loading dataset...")
            for row in rd:
                #print row
                self.data.append(self.__preprocess_row(row))
                #print self.data[-1]

    def __preprocess_row(self, row):
        res = []
        for idx in range(0, len(row)):
            tname = self.header_type[self.header_row[idx]]
            tdata = row[idx]
            if tdata == '?':
                tdata = None
            if tname == NOMINAL:
                res.append(tdata)
            elif tname == NUMERIC:
                if tdata:
                    tdata = float(tdata)
                res.append(tdata)
            else:
                print("ERROR: UKNOW HEADER => "+tname)
                sys.exit(0)
        return res

    def __postprocess_row(self, row):
        res = []
        for item in row:
            if item != None:
                res.append(item)
            else:
                res.append('?')
        return res
        
    def save_dataset(self, outfile):
        with open(outfile, 'w') as csvfile:
            print("Saving dataset to "+outfile+"...")
            wr = csv.writer(csvfile, delimiter=',')
            wr.writerow(self.header_row)
            for row in self.data:
                wr.writerow(self.__postprocess_row(row))

    def train_model(self, prune):
        for row in self.data:
            val = row[self.header_index[self.target]]
            if not(val in self.target_values) and val != None:
                self.target_values.append(val)
        print "target values: ",self.target_values
        input_data = []
        count_map = {}
        for tt in self.target_values:
            count_map[tt] = 0
        for row in self.data:
            target = row[self.header_index[self.target]]
            if target != None:
                input_data.append(row)
                count_map[target] += 1
        dtree = DTree()
        dtree.root = {}
        self.__build_tree(dtree.root, input_data)
        return dtree

    def __calc_entropy(self, count_map):
        entropy = 0.0
        count = 0
        #print(count_map)
        for (key, val) in count_map.items():
            count += val
        if count == 0:
            print("entrop == 0 ??!!!")
            sys.exit(0)
        count = float(count)
        for (key, val) in count_map.items():
            if val == 0:
                continue
            p = val / count
            #print("proba => "+str(p))
            entropy += -p*math.log(p, 2)
        #print("entrop => "+str(entropy))
        return entropy

    def __build_tree(self, root, dataset):
        if self.__is_pure(dataset):
            root["leaf"] = True
            root["label"] = dataset[0][self.header_index[self.target]]
            print("is pure => ")
            return
        print "continue"
        #print dataset
        ent_count_map = {}
        for tt in self.target_values:
            ent_count_map[tt] = 0
        for row in dataset:
            target = row[self.header_index[self.target]]
            ent_count_map[target] += 1
        last_entropy = self.__calc_entropy(ent_count_map)
        print("last-entropy => "+str(last_entropy))
        # need to branch again
        # probe is actually a dict has keys {title, gain, values => [], entropy}
        root["leaf"] = False
        root["subtree"] = []
        probe_list = []
        for title in self.header_row:
            if title != self.target:
                probe_item = self.__probe_title(title, dataset, last_entropy)
#                print probe_item
                probe_list.append(probe_item)

        assert(len(probe_list) > 0)
        max_gain = probe_list[0]["gain"]
        max_candidate = probe_list[0]
        for candidate in probe_list:
            if max_gain < candidate["gain"]:
                max_gain = candidate["gain"]
                max_candidate = candidate
        print max_candidate
        if max_candidate["gain"] < 0.00001:
            root["leaf"] = True
            tidx = self.header_index[self.target]
            target_count = {}
            for tt in self.target_values:
                target_count[tt] = 0
            for row in dataset:
                target_count[row[tidx]] += 1
            root["label"] = max(target_count, key=target_count.get)
            return
        # if it is still a leaf
        root["name"] = max_candidate["title"]
        root["data_type"] = self.header_type[max_candidate["title"]]
        root["mean_value"] = max_candidate["mean_value"]
        root["marjor_value"] = max_candidate["marjor_value"]

        # TODO: construct the subtrees
        root["subtree"], split_dataset = self.__split_probe(max_candidate, dataset)
        for idx in range(0, len(root["subtree"])):
            self.__build_tree(root["subtree"][idx]["tree"], split_dataset[idx])

    def __is_pure(self, dataset):
        res  = {}
        for row in dataset:
            val = row[self.header_index[self.target]]
            if val != None:
                res[val] = True
        return len(res) == 1

    def __probe_title(self, title, dataset, last_entropy):
        res = {"title":title}
        dat = []
        vals = []
        val_count = {}
        missing_row = []
        target_count = {}
        dtype = self.header_type[title]
        if dtype == NUMERIC:
            for tt in self.target_values:
                val_count[tt] = 0
        for row in dataset:
            val = row[self.header_index[title]]
            target = row[self.header_index[self.target]]
            if val != None and target != None:
                dat.append(row[self.header_index[title]])
                if dtype == NOMINAL:
                    if val in val_count:
                        val_count[val] += 1
                    else:
                        val_count[val] = 1
                        vals.append(val)
                        target_count[val] = {}
                        for tt in self.target_values:
                            target_count[val][tt] = 0
                    target_count[val][target] += 1
                elif dtype == NUMERIC:
                    vals.append(val)
                    if val in target_count:
                        target_count[val][target] += 1
                    else:
                        target_count[val] = {}
                        for tt in self.target_values:
                            target_count[val][tt] = 0
                        target_count[val][target] += 1
                    val_count[target] += 1
                else:
                    print("UNKNOWN data type: "+dtype)
                    sys.exit(1)
            else:
                missing_row.append(row)
        if dtype == NOMINAL:
#            if title == "weather":
#                print "stat => val_count", val_count, "target_count", target_count
            sum_sum = 0.0
            ent_sum = 0.0
            if len(val_count) == 0:
                res["entropy"] = last_entropy
                res["gain"] = 0
                print title
                print val_count
                print dataset
                return res
            mval = max(val_count, key=val_count.get)
            for row in missing_row:
                target = row[self.header_index[self.target]]
                val_count[mval] += 1
                target_count[mval][target] += 1
            for (key, val) in val_count.items():
                sum_sum += val
            for (key, val) in val_count.items():
                ent_sum += self.__calc_entropy(target_count[key]) * val / sum_sum
            #split_ent = self.__calc_entropy(val_count)
            split_ent = 1
            res["entropy"] = ent_sum
            res["gain"] = (last_entropy - ent_sum) / split_ent
            res["mean_value"] = mval
            res["marjor_value"] = mval
            res["values"] = list(val_count.keys())
        elif dtype == NUMERIC:
            acc = {}
            for tt in self.target_values:
                acc[tt] = 0
            means_val = 0
            for item in vals:
                means_val += item
            # if no non-None value exists
            if len(vals) == 0:
                print(dataset)
                res["entropy"] = last_entropy
                res["gain"] = 0.0
                return res
                #sys.exit(1)
            means_val = means_val / float(len(vals))
            vals.append(means_val)
            for row in missing_row:
                val_count[row[self.header_index[self.target]]] += 1
            if not (means_val in target_count):
                target_count[means_val] = {}
                for tt in self.target_values:
                    target_count[means_val][tt] = 0
            for row in missing_row:
                target_count[means_val][row[self.header_index[self.target]]] += 1
            val_set = set(vals)
            vals = list(val_set)
            vals.sort()
            min_mid_val = None
            min_gain = None
            min_entropy = None
            for idx in range(0, len(vals)-1):
                prev_val = vals[idx]
                next_val = vals[idx + 1]
                mid_val = (prev_val + next_val) / 2.0
                prev_sum = 0.0
                next_sum = 0.0
                for tt in self.target_values:
                    acc[tt] += target_count[prev_val][tt]
                    val_count[tt] -= target_count[prev_val][tt]
                    prev_sum += acc[tt]
                    next_sum += val_count[tt]
                prev_ent = self.__calc_entropy(acc)
                next_ent = self.__calc_entropy(val_count)
                #split_ent = self.__calc_entropy({1:prev_sum, 2:next_sum})
                split_ent = 1
                sum_sum = prev_sum + next_sum
                cur_entropy = prev_ent * prev_sum / sum_sum + next_ent * next_sum / sum_sum
                gain = (last_entropy - cur_entropy) / split_ent
                if min_gain == None or min_gain < gain:
                    min_gain = gain
                    min_mid_val = mid_val
                    min_entropy = cur_entropy
#                    if title == "oppnuminjured":
#                        print ">>>>>>>>>>>>>",split_ent, [prev_sum,next_sum], vals, prev_val, next_val, mid_val
            res["gain"] = min_gain
            res["entropy"] = min_entropy
            res["values"] = [min_mid_val]
            res["mean_value"] = means_val
            res["marjor_value"] = means_val
        else:
            print("unknown data type"+dtype)
            sys.exit(1)
        return res

    def __split_probe(self, probe, dataset):
        split_res = []
        split_dataset = []
        dtype = self.header_type[probe["title"]]
        title_idx = self.header_index[probe["title"]]
        if dtype == NOMINAL:
            val_map = {}
            for val in probe["values"]:
                insert_map = {"ops":CMP_EQ, "value":val, "tree":{}}
                insert_dataset = []
                split_res.append(insert_map)
                split_dataset.append(insert_dataset)
                val_map[val] = {"map":insert_map,"dataset":insert_dataset}
            for row in dataset:
                val = row[title_idx]
                if val == None:
                    val = probe["mean_value"]
                val_map[val]["dataset"].append(row)
        elif dtype == NUMERIC:
            tv = probe["values"][0]
            split_res.append({"ops":CMP_LT,"value":tv,"tree":{}})
            split_res.append({"ops":CMP_GE,"value":tv,"tree":{}})
            split_dataset = [[],[]]
            for row in dataset:
                val = row[title_idx]
                if val == None:
                    val = probe["mean_value"]
                if val < tv:
                    split_dataset[0].append(row)
                else:
                    split_dataset[1].append(row)
        else:
            print("UNKNOWN data type: "+ dtype)
            sys.exit(1)
        return split_res, split_dataset

    def predict_one(self, row):
        pass

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
            print model.root
            if args["print"]:
                mode.print_model()
            if len(args["model"]) > 0 :
                model.save_model(args["model"])
            if len(args["output"]) > 0 :
                dataset.save_dataset(args["output"])
        else:
            print("wrong input file: "+args["input"])
    elif args["action"] == "validate":
        if os.path.isfile(args["input"]) and os.path.isfile(args["model"]):
            print("Do validation")
            model = DTree()
            model.load_model(args["model"])
        else:
            print("wrong input file: "+args["input"]+"or wrong model file: "+args["model"])
        pass
    elif args["action"] == "predict":
        if os.path.isfile(args["input"]) and os.path.isfile(args["model"]):
            print("Do prediction")
            model = DTree()
            model.load_model(args["model"])
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
    try:
        main(params)
    except RuntimeError:
        print("hehe")

