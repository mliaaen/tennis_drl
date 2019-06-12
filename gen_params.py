import json

# Helper functions

def get_hyperparams(**args) -> object:
    import itertools
    param_list = []
    field_list = []
    for k, v in args.items():
        #print("%s = %s" % (k, v))
        param_list.append(v)
        field_list.append(k)

    config_list = []
    for i in itertools.product(*param_list):
        config_list.append(dict(zip(field_list, i)))

    return field_list, config_list




############## running hyper parameter searching ######################


parameters_best = {"n_episodes" :[2500],
              "seeds":[1],
              " lr_critic":[1e-3],
              "lr_actor":[1e-4],
              "weight_deacy":[1],
              "learn_num":[5],
              "gamma":[0.99],
              "tau":[7e-2],
              "ou_sigma":[0.3],
              "ou_theta":[0.15],
              "eps_start":[1.0],
              "eps_ep_end":[400],
              "eps_final":[0.1],
              "batch_size": [96, 128, 512],

              "hidden_units": [64, 96, 128]
              }


parameters = {"n_episodes" :[2000],
            "seeds":[1],
            "lr_critic":[1e-3, 1e-4],
            "lr_actor":[1e-3, 1e-4],
            "learn_every": [5],
            "hidden_units": [128]
              }

import argparse

if __name__ == '__main__':

    description = """
Generate run configurations
"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', action="store", dest="scenarios", default=None, help="The json formatted template")
    parser.add_argument('--output', action="store", dest="output", default="", help="The resulting configs json file")

    args = parser.parse_args()


    with open(args.scenarios,"r") as f:
        scenarios = json.load(f)
        field_list, configs = get_hyperparams(**scenarios)

    with open(args.output, "w") as write_file:
        json.dump(configs, write_file, indent=4)
