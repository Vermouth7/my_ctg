import argparse
import json
import os


def check_train(datum: dict, unseen_combs: list) -> bool:
    for i in range(len(unseen_combs)):
        flag = True
        for key in unseen_combs[i].keys():
            if datum[key] != unseen_combs[i][key]:
                flag = False
        if flag == False:  # at least one attribute different
            pass
        else:  # in this combination, all the attributes match
            return False
    return True

def get_data_by_unseen_combs(dataset_path: str, unseen_combs: list) -> list:
    with open(dataset_path, 'r') as f:
        all_data = [json.loads(item) for item in f.readlines()]

    data_train = [datum for datum in all_data if check_train(datum, unseen_combs)]
    return data_train

def get_train_dataset(dataset_path: str, unseen_combs_path: str, mode: str, idx: int) -> list:
    unseen_combs_dict = {}
    with open(unseen_combs_path, 'r') as f:
        for item in f.readlines():
            dic = json.loads(item)
            unseen_combs = dic['unseen_combs']
            _idx = dic['idx']
            _mode = dic['mode']
            if _mode not in unseen_combs_dict:
                unseen_combs_dict[_mode] = []
            unseen_combs_dict[_mode].append((unseen_combs, _mode, _idx))

    assert mode in unseen_combs_dict
    assert idx < len(unseen_combs_dict[mode])

    unseen_combs = unseen_combs_dict[mode][idx][0]
    train_dataset = get_data_by_unseen_combs(dataset_path=dataset_path, unseen_combs=unseen_combs)
    mode_name = unseen_combs_dict[mode][idx][1] + str(unseen_combs_dict[mode][idx][2])

    return train_dataset, mode_name, unseen_combs

def create_new_data(train_dataset: list, unseen_combs: list) -> list:
    new_data = []
    for datum in train_dataset:
        attributes = ', '.join([f"{key}: {datum[key]}" for key in datum if key != 'review'])
        instruction = f"Write a sentence that meets the requirement of input control conditions: {attributes}"
        output_data = {
            "instruction": instruction,
            "output": datum['review'],
            'sentiment': datum['sentiment'],
            'pronoun': datum['pronoun'],
            'tense': datum['tense']
        }
        new_data.append(output_data)
    return new_data

def save_data(new_data: list, mode_name: str):
    output_dir = "./sft/train_cg4mctg"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{mode_name}.json")
    with open(file_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/home/chh/repos/CG4MCTG/data/Yelp/gen.jsonl", type=str)
    parser.add_argument("--unseen_combs_path", default="/home/chh/repos/CG4MCTG/data/Yelp/unseen.jsonl", type=str)
    args = parser.parse_args()

    modes = ['Original', 'Hold-Out', 'ACD', 'Few-Shot']
    mode_indices = {
        'Hold-Out': 8,
        'ACD': 10,
        'Few-Shot': 8,
        'Original': 1  # Assuming there's only one Original mode
    }

    for mode in modes:
        for idx in range(mode_indices[mode]):
            train_dataset, mode_name, unseen_combs = get_train_dataset(dataset_path=args.dataset_path, unseen_combs_path=args.unseen_combs_path, mode=mode, idx=idx)
            new_data = create_new_data(train_dataset, unseen_combs)
            save_data(new_data, mode_name)
