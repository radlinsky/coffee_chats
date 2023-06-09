import argparse
import itertools
import pandas as pd
from parse_inputs import parse_windows_outlook_tracking

def main(parsed_args):
    if parsed_args.outlook_tracking:
        names = parse_windows_outlook_tracking(parsed_args.names_file)
    else:
        names = []
        with open(parsed_args.names_file, "r") as handle:
            for line in handle:
                name = line.strip("\n\r")
                names.append(name)
    
    names.sort()

    if parsed_args.verbose:
        print(f"{len(names)} names parsed:\n{names}")
        
    weights = pd.read_csv(parsed_args.weights_file, sep = "\t")
    check_completeness(weights)

    if parsed_args.verbose:
        print(f"generating every way to split up {len(names)} into {parsed_args.group_size} groups")
    every_splitup_group = generate_every_splitup_group(names, parsed_args.group_size)
    if parsed_args.verbose:
        print(f"{len(every_splitup_group)} ways to split up names found.")
        print("getting min weight grouping...")
    weight_dict = dict(zip(weights.combo, weights.weight))
    best_grouping = get_minimum_weight_grouping(every_splitup_group, weight_dict)
    
    # add 1 to each weights for each combo in best_grouping
    updated_weights = add_one_weight(weight_dict, {k: weight_dict[k] for k in best_grouping})
    
    return({"groups": best_grouping, "weights": updated_weights})

def add_one_weight(current_weight_df, weights_to_add_one):
    thesecombos = []
    theseweights = []
    for k,v in weights_to_add_one.items():
        thesecombos.append(k)
        theseweights.append(v + 1)
    weights_plus1_df = pd.DataFrame({"combo":thesecombos,"weight":theseweights})
    updated_weights = update_weight_matrix(current_weight_df, weights_plus1_df)
    return updated_weights


def email_friendly(groups):
    pass
    
    
def every_way_to_split_up(names, n):
    """
    Given list and a number, find every way to split up the list into groups of size n
    
    names modulo n MUST be 0
    """
    if not names:
        yield []
    else:
        for group in (((names[0],) + xs) for xs in itertools.combinations(names[1:], n-1)):
            for groups in every_way_to_split_up([x for x in names if x not in group], n):
                yield [group] + groups

                
def generate_every_splitup_group(names, n, verbose = False):
    thenames = names.copy()
    # len of names needs to be perfectly divisible by n
    # add "" to the list of names so len modulo n is 0
    while len(thenames) % n != 0:
        if verbose:
            print("padded thenames")
        thenames = thenames + [""]
    
    return list(every_way_to_split_up(thenames, n))


def get_minimum_weight_grouping(list_of_groups, weights, verbose = False):
    """
    list_of_groups: list of lists of sets
        e.g. [[("A","B"),("C","D")], [("A","C"),("B","D")], ...]
    weights: dict, set -> numeric weight
        e.g. {("A","B"): 3, ("C","D"): 1, ("A","C"): 3, ("B","D"): 0, ...}
        
    For each list of sets, get the weights for each set an dsum them. Return the list of
        sets with the smallest sum.
    """
    # set initial weight_sum to beat at maximum possible value if one big group
    weight_sum = sum(weights.values())
    if verbose:
        print(weight_sum)
    # set initial group to the first group
    best_group = list_of_groups[0]
    for this_group in list_of_groups:
        this_weight_sum = sum([weights[combo] for combo in this_group])
        if this_weight_sum < weight_sum:
            weight_sum = this_weight_sum
            if verbose:
                print(weight_sum)
            best_group = this_group
    return best_group


def initialize_weight_matrix(names):
    names.sort()
    combos  = [c for c in itertools.combinations(names+[""], 2)]
    weights = pd.DataFrame(
        {
            "combo":combos,
            "weight":[0 for n in combos]
        }
    )
    return weights


def check_completeness(weight_df):
    # ensure that every combination of names exists in the weight_df 
    names = []
    combos = []
    for c in weight_df["combo"]:
        names.extend(c)
        combos.append(c)
    
    names = list(set(names))
    names.sort()
    names.remove("")
    every_combo = [c for c in itertools.combinations(names+[""], 2)]
    for c in every_combo:
        if c not in combos:
            raise ValueError(f"missing combo {c}!!")


def update_weight_matrix(current_weight_df, new_weight_df):
    # replace weights of current df with weights in new df
    # adds new rows if there are new combos (make sure every combination of names is accounted for!)
    existing_combos = new_weight_df.loc[new_weight_df.combo.isin(current_weight_df.combo)]
    current_weight_df.loc[
        current_weight_df.combo.isin(existing_combos.combo),
        ['weight']
    ] = existing_combos[['weight']]
    # add new combos as new rows
    new_combos = new_weight_df.loc[~new_weight_df.combo.isin(current_weight_df.combo)]
    updated = pd.concat([current_weight_df, new_combos])
    # careful! make sure every combo is accounted for!
    check_completeness(updated)
    return updated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("names_file", 
                        help = "Text file with names, each name on a new line"
                       )
    parser.add_argument("weights_file", 
                        help = "Tab delim file with 3 cols: every pair of names and a weight."
                       )
    parser.add_argument("--outlook_tracking", 
                        help = "names_file comes from copy pasting outlook file",
                        action = "store_true"
                        default = False
                       )
    parser.add_argument("--group_size",
                       default = 2)
    args = parser.parse_args()
    if args.verbose:
        print(args.echo)
    results = main(args)
    
    i = 1
    for combo in results["groups"]:
        print(f"Group {i}: {' and '.join(list(combo))}")
        i += 1

    new_weights_file = args.weight_file + ".updated"
    results["weights"].to_csv(new_weight_file)
    print(f"updated weights:\n{new_weights_file}\n")