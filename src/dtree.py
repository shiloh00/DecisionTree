#!/usr/bin/python

import csv
import argparse
import os
import sys
import math
import json
import random

NUMERIC = "numeric"
NOMINAL = "nominal"

CMP_LT = "<"
CMP_EQ = "=="
CMP_GE = ">="

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

    # print the model in DNF format
    def print_model(self):
        if self.root == None:
            print("None model")
        else:
            plist = []
            self.__print_node(self.root, plist)

    def __print_node(self, node, plist):
        if node["leaf"]:
            print(" AND ".join(plist)+" => "+node["label"])
        else:
            for subtree in node["subtree"]:
                val = subtree["value"]
                if type(val) is float:
                    val = "%.3f" % val
                tstr = node["name"]+" "+subtree["ops"]+" "+val
                plist.append(tstr)
                self.__print_node(subtree["tree"], plist)
                plist.pop()
            


class Dataset:
    """stand for a loaded dataset"""

    build_tree_count = 0
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

    validate_data = []

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
            #print(self.header_row)
            self.data = []
            print("Loading dataset...")
            for row in rd:
                #print row
                self.data.append(self.__preprocess_row(row))
                #print self.data[-1]

    def load_validate(self, path):
        with open(path, 'rb') as csvfile:
            print("Loading validate dataset")
            rd = csv.reader(csvfile, delimiter=',')
            __header_row = next(rd, None)
            tidx = self.header_index[self.target]
            for row in rd:
                out_row = self.__preprocess_row(row)
                target = out_row[tidx]
                if target != None:
                    self.validate_data.append(out_row)
            print("Loaded "+str(len(self.validate_data))+" entries for validation from "+path)

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
        #print self.data
        return self.__train_model(self.data, prune)

    def __prune_tree(self, model, validate_data):
        print("Using validation set("+str(len(validate_data))+") to prune")
        tidx = self.header_index[self.target]
        corr_count = 0
        for row in validate_data:
            target = row[tidx]
            predict_val, chain = self.predict_one(model, row)
            if predict_val == target:
                corr_count += 1
            for node in chain:
                node["prune_meta"]["access_count"] += 1
                if target == node["label"]:
                    node["prune_meta"]["correct_count"] += 1
        print("before pruning accuracy => " + str(corr_count/float(len(validate_data))))
        self.__prune_tree_node(model.root)
        corr_count = 0
        for row in validate_data:
            target = row[tidx]
            predict_val, chain = self.predict_one(model, row)
            if predict_val == target:
                corr_count += 1
        print("after pruning accuracy => " + str(corr_count/float(len(validate_data))))


    def __prune_tree_node(self, node):
        if node["leaf"]:
            return
        for subtree in node["subtree"]:
            self.__prune_tree_node(subtree["tree"])

        corr_rate = 0.01
        if node["prune_meta"]["access_count"] != 0:
            corr_rate = node["prune_meta"]["correct_count"]/float(node["prune_meta"]["access_count"])
        child_corr = 0
        child_access = 0.0
        non_leaf = False
        label_set = {}
        for subtree in node["subtree"]:
            child_corr += subtree["tree"]["prune_meta"]["correct_count"]
            child_access += subtree["tree"]["prune_meta"]["access_count"]
            if subtree["tree"]["leaf"]:
                label_set[subtree["tree"]["label"]] = True
            else:
                non_leaf = True

        if not non_leaf:
            child_rate = 0.0
            if child_access > 0.1:
                child_rate = child_corr / float(child_access)
            if corr_rate > child_rate:
                self.build_tree_count -= len(node["subtree"])
                node["subtree"] = []
                node["leaf"] = True
            else:
                if len(label_set) == 1:
                    self.build_tree_count -= len(node["subtree"])
                    node["leaf"] = True
                    node["label"] = node["subtree"][0]["tree"]["label"]
                    node["subtree"] = []
                node["prune_meta"]["correct_count"] = child_corr
                node["prune_meta"]["access_count"] = child_access
        else:
            child_rate = 0.0
            if child_access > 0.1:
                child_rate = child_corr / float(child_access)
            if corr_rate > child_rate:
                self.build_tree_count -= len(node["subtree"])
                node["subtree"] = []
                node["leaf"] = True
            else:
                node["prune_meta"]["correct_count"] = child_corr
                node["prune_meta"]["access_count"] = child_access


            

    def __train_model(self, dataset, prune):
        self.build_tree_count = 0
        for row in dataset:
            val = row[self.header_index[self.target]]
            if not(val in self.target_values) and val != None:
                self.target_values.append(val)
        input_data = []
        count_map = {}
        for tt in self.target_values:
            count_map[tt] = 0
        for row in dataset:
            target = row[self.header_index[self.target]]
            if target != None:
                input_data.append(row)
                count_map[target] += 1
        dtree = DTree()
        dtree.root = {}
        self.__build_tree(dtree.root, input_data)
        print("train done")
        print("Generated "+str(self.build_tree_count)+" nodes")
        if prune:
            print("begin to prune the generated tree...")
            self.__prune_tree(dtree, self.validate_data)
            print("pruning tree done")
            print("After pruning, there are "+str(self.build_tree_count)+" nodes")
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
        self.build_tree_count += 1
        root["prune_meta"] = {"correct_count":0,"access_count":0}
        for tt in self.target_values:
            root["prune_meta"][tt] = 0
        if self.__is_pure(dataset):
            root["leaf"] = True
            root["label"] = dataset[0][self.header_index[self.target]]
            #print("is pure => ")
            return
        #print dataset
        ent_count_map = {}
        for tt in self.target_values:
            ent_count_map[tt] = 0
        for row in dataset:
            target = row[self.header_index[self.target]]
            ent_count_map[target] += 1
        last_entropy = self.__calc_entropy(ent_count_map)
        root["label"] = max(ent_count_map, key=ent_count_map.get)
        if self.__is_any_pure(dataset):
            root["leaf"] = True
            return
        # need to branch again
        # probe is actually a dict has keys {title, gain, values => [], entropy}
        root["leaf"] = False
        root["subtree"] = []
        probe_list = []
        for title in self.header_row:
            if title != self.target:
                probe_item = self.__probe_title(title, dataset, last_entropy)
                probe_list.append(probe_item)

        assert(len(probe_list) > 0)
        max_gain_ratio = probe_list[0]["gain_ratio"]
        max_candidate = probe_list[0]
        for candidate in probe_list:
            if max_gain_ratio < candidate["gain_ratio"]:
                max_gain_ratio = candidate["gain_ratio"]
                max_candidate = candidate
        #print max_candidate
        #print dataset
        if max_candidate["gain_ratio"] < 0.03:
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

    def __is_any_pure(self, dataset):
        tidx = self.header_index[self.target]
        for title in self.header_row:
            title_idx = self.header_index[title]
            if title_idx != tidx:
                res = {}
                for row in dataset:
                    val = row[title_idx]
                    if val != None:
                        res[row[tidx]] = True
                if len(res) <= 1:
                    return True
        return False

    def __probe_title(self, title, dataset, last_entropy):
        res = {"title":title}
        dat = []
        vals = []
        val_count = {}
#        missing_row = []
        target_count = {}
        dtype = self.header_type[title]
        if dtype == NUMERIC:
            for tt in self.target_values:
                val_count[tt] = 0
        for row in dataset:
            val = row[self.header_index[title]]
            target = row[self.header_index[self.target]]
            #label_count[tt] += 1
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
#            else:
#                missing_row.append(row)
        if dtype == NOMINAL:
            sum_sum = 0.0
            ent_sum = 0.0
            # in case all instances miss this attribute
            if len(val_count) == 0 or len(val_count) == 1:
                res["entropy"] = last_entropy
                res["gain_ratio"] = 0
                return res
            mval = max(val_count, key=val_count.get)
            #print val_count
#            if len(val_count) == 1:
#                res["entropy"] = 0
#                res["gain"] = last_entropy
#                res["mean_value"] = mval
#                res["marjor_value"] = mval
#                res["values"] = [mval]
#                return
#            for row in missing_row:
#                target = row[self.header_index[self.target]]
#                val_count[mval] += 1
#                target_count[mval][target] += 1
            for (key, val) in val_count.items():
                sum_sum += val
            for (key, val) in val_count.items():
                ent_sum += self.__calc_entropy(target_count[key]) * val / sum_sum
            split_ent = 1
            res["entropy"] = ent_sum
            res["gain_ratio"] = (last_entropy - ent_sum) / split_ent
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
                res["entropy"] = last_entropy
                res["gain_ratio"] = 0.0
                return res
            means_val = means_val / float(len(vals))
            vals.append(means_val)
#            for row in missing_row:
#                val_count[row[self.header_index[self.target]]] += 1
            if not (means_val in target_count):
                target_count[means_val] = {}
                for tt in self.target_values:
                    target_count[means_val][tt] = 0
#            for row in missing_row:
#                target_count[means_val][row[self.header_index[self.target]]] += 1
            val_set = set(vals)
            vals = list(val_set)
            vals.sort()
            min_mid_val = None
            min_gain_ratio = None
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
                split_ent = 1
                #split_ent = self.__calc_entropy({1:prev_sum,2:next_sum})
                sum_sum = prev_sum + next_sum
                cur_entropy = prev_ent * prev_sum / sum_sum + next_ent * next_sum / sum_sum
                gain_ratio = (last_entropy - cur_entropy) / split_ent
                if min_gain_ratio == None or min_gain_ratio < gain_ratio:
                    min_gain_ratio = gain_ratio
                    min_mid_val = mid_val
                    min_entropy = cur_entropy
            res["gain_ratio"] = min_gain_ratio
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

    def __accept(self, val, ops, target):
        if ops == CMP_EQ:
            return val == target
        elif ops == CMP_LT:
            return val < target
        elif ops == CMP_GE:
            return val >= target
        else:
            print("UNKNOWN ops: "+ops)
            sys.exit(1)

    def predict_one(self, model, row):
        root = model.root
        chain = [root]
        while not root["leaf"]:
            title_idx = self.header_index[root["name"]]
            val = row[title_idx]
            if val == None:
                val = root["mean_value"]
            found = False
            for subtree in root["subtree"]:
                if self.__accept(val, subtree["ops"], subtree["value"]):
                    root = subtree["tree"]
                    found = True
                    chain.append(root)
                    break
            if not found:
                break
                #print val
                #sys.exit(1)
        return root["label"], chain

    def test_all(self, model):
        return self.test(model, self.data)

    def test(self, model, test_data):
        correct_count = 0
        error_count = 0
        tidx = self.header_index[self.target]
        for row in test_data:
            target = row[tidx]
            if target != None:
                predict_val, chain = self.predict_one(model, row)
                if target == predict_val:
                    correct_count += 1
                else:
                    error_count += 1
        total_count = float(correct_count + error_count)
        print("accuracy => "+str(correct_count/total_count)+" ("+str(correct_count)+"/"+str(correct_count+error_count)+")")
        return correct_count/total_count

    def predict(self, model):
        pass

    def __split_fold(self, K):
        folds = []
        tmp_data = []
        tidx = self.header_index[self.target]
        for row in self.data:
            target = row[tidx]
            if target != None:
                tmp_data.append(row)
        random.shuffle(tmp_data)
        num = len(tmp_data)/K
        for idx in range(0, K):
            if idx == K-1:
                folds.append(tmp_data[num*idx:])
            else:
                folds.append(tmp_data[num*idx:num*(idx+1)])
        return folds


    def cross_validate(self, K, prune):
        acc = []
        folds = self.__split_fold(K)

        for idx in range(0, K):
            train_set = []
            test_set = folds[idx]
            print("done split => "+str(idx))
            for ii in range(0, K):
                if ii != idx:
                    train_set += folds[ii]
            acc.append(self.test(self.__train_model(train_set, prune), test_set))

        print(acc)
        print("average accuracy => "+str(sum(acc)/float(K)))


def main(args):
    print(args)
    if args["action"] == "train":
        print("Do training")
        if os.path.isfile(args["input"]):
            dataset = Dataset(args["input"])
            if args["prune"]:
                if not os.path.isfile(args["validate"]):
                    print("Pruning is specified but no validation set is provided")
                    sys.exit(1)
                else:
                    dataset.load_validate(args["validate"])
            model = dataset.train_model(args["prune"])
            #print model.root
            if args["print"]:
                model.print_model()
            if len(args["model"]) > 0 :
                model.save_model(args["model"])
            if len(args["output"]) > 0 :
                dataset.save_dataset(args["output"])
            if args["print"]:
                model.print_model()
        else:
            print("wrong input file: "+args["input"])
    elif args["action"] == "validate":
        if os.path.isfile(args["input"]):
            print("Do validation")
            dataset = Dataset(args["input"])
            if os.path.isfile(args["model"]):
                model = DTree()
                model.load_model(args["model"])
                print("Validate using provided model...")
                print(dataset.test_all(model))
            else:
                print("Validate with 10-fold cross-validation")
                if args["prune"]:
                    if not os.path.isfile(args["validate"]):
                        print("Pruning is specified but no validation set is provided")
                        sys.exit(1)
                    else:
                        dataset.load_validate(args["validate"])
                dataset.cross_validate(10, args["prune"])
        else:
            print("wrong input file: "+args["input"]+" or wrong model file: "+args["model"])
    elif args["action"] == "predict":
        if os.path.isfile(args["input"]) and os.path.isfile(args["model"]):
            print("Do prediction")
            model = DTree()
            model.load_model(args["model"])
        else:
            print("wrong input file: "+args["input"]+" or wrong model file: "+args["model"])
    else:
        print("UNKNOWN ACTION: " + args["action"])
    print("Done.")


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser(description="Train, test or validate the input dataset using C4.5 decision tree algorithm")
    opt_parser.add_argument('-a', '--action', dest='action', type=str, default='train', choices=['train','validate','predict'], required=True, help='specify the action')
    opt_parser.add_argument('-i', '--input', dest='input', type=str, default='', help='specify the input file(dataset)')
    opt_parser.add_argument('-v', '--validate-dataset', dest='validate', type=str, default='', help='specify the validation file(dataset)')
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

