import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch
import gym

import discrete_BCQ
import time
import BCQ_utils
import atari_BCQ


if __name__ == "__main__":
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    atari_parameters = {
        # Exploration
        "start_timesteps": 2e4,
        "initial_eps": 1,
        "end_eps": 1e-2,
        "eps_decay_period": 25e4,
        # Evaluation
        "eval_freq": 5e4,
        "eval_eps": 1e-3,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 32,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.0000625,
            "eps": 0.00015
        },
        "train_freq": 4,
        "polyak_target_update": False,
        "target_update_freq": 8e3,
        "tau": 1
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="AsterixNoFrameskip-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Default")  # Prepends name to filename
    parser.add_argument("--max_timesteps", default=5e6, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--BCQ_threshold", default=0.3, type=float)  # Threshold hyper-parameter for BCQ
    parser.add_argument("--low_noise_p", default=0.2,
                        type=float)  # Probability of a low noise episode when generating buffer
    parser.add_argument("--rand_action_p", default=0.2,
                        type=float)  # Probability of taking a random action when generating buffer, during non-low noise episode
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral policy
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    args = parser.parse_args()

    args.train_behavioral = True

    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    # if not os.path.exists("./results"):
    #     os.makedirs("./results")

    if not os.path.exists("./../models"):
        os.makedirs("./../models")

    if not os.path.exists("./../buffers"):
        os.makedirs("./../buffers")

    # Make env and determine properties
    env, is_atari, state_dim, num_actions = BCQ_utils.make_env(args.env, atari_preprocessing)
    parameters = atari_parameters

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = BCQ_utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"],
                                       parameters["buffer_size"], device)

    if args.train_behavioral or args.generate_buffer:
        atari_BCQ.interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
    else:
        atari_BCQ.train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)