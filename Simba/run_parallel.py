import argparse
import copy
import multiprocessing as mp
import os
import time

import numpy as np


def run_with_device(server, device_id, config_path, config_name, overrides):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    os.environ["OMP_NUM_THREADS"] = "2"

    # Now import the main script
    if config_name == "online_rl":
        from run_online import run
    else:
        raise NotImplementedError

    args = {
        "config_path": config_path,
        "config_name": config_name,
        "overrides": overrides,
    }
    run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="online_rl")
    parser.add_argument("--agent_config", type=str, default="simbaV2")
    parser.add_argument("--env_type", type=str, default="dmc_hard")
    parser.add_argument("--device_ids", default=[0], nargs="+")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--num_exp_per_device", type=int, default=1)
    parser.add_argument("--server", type=str, default="local")
    parser.add_argument("--group_name", type=str, default="test")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--overrides", action="append", default=[])

    args = vars(parser.parse_args())
    seeds = (np.arange(args.pop("num_seeds")) * 1).tolist()
    device_ids = args.pop("device_ids")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    num_devices = len(device_ids)
    num_exp_per_device = args.pop("num_exp_per_device")
    pool_size = num_devices * num_exp_per_device

    # create configurations for child run
    experiments = []
    config_path = args.pop("config_path")
    config_name = args.pop("config_name")
    server = args.pop("server")
    group_name = args.pop("group_name")
    exp_name = args.pop("exp_name")
    agent_config = args.pop("agent_config")

    # import library after CUDA_VISIBLE_DEVICES operation
    from scale_rl.envs.dmc import DMC_EASY_MEDIUM, DMC_HARD, HUM_STAND, HUM_WALK, HUM_RUN, DOG_STAND, DOG_WALK, DOG_TROT, DOG_RUN
    from scale_rl.envs.humanoid_bench import HB_LOCOMOTION_NOHAND, HB_WALK, HB_CRAWL, HB_STAND, HB_RUN, HB_HURDLE, HB_MAZE, HB_STAIR, HB_POLE, HB_SLIDE, HB_SIT1, HB_SIT2, HB_BALANCE1, HB_BALANCE2
    from scale_rl.envs.humanoid_bench import HBH_WALK, HBH_CRAWL, HBH_STAND, HBH_RUN, HBH_HURDLE, HBH_MAZE, HBH_STAIR, HBH_POLE, HBH_SLIDE, HBH_REACH, HBH_BAL_H, HBH_BAL_S, HBH_SIT_H, HBH_SIT_S
    from scale_rl.envs.humanoid_bench import HBH_BASKET, HBH_BOOK_H, HBH_BOOK_S, HBH_CABINET, HBH_CUBE, HBH_DOOR, HBH_INSERT_N, HBH_INSERT_S, HBH_POWERLIFT, HBH_PUSH, HBH_ROOM, HBH_SPOON, HBH_TRUCK, HBH_WINDOW
    from scale_rl.envs.myosuite import MYOSUITE_TASKS, MYO1, MYO2, MYO3, MYO4, MYO5, MYO6, MYO7, MYO8, MYO9, MYO10
    from scale_rl.envs.mujoco import MUJOCO_ALL, MUJ_Cheetah, MUJ_Humanoid, MUJ_Hopper, MUJ_Walker, MUJ_Ant

    env_type = args.pop("env_type")

    ###################
    # online

    if env_type == "dmc_em":
        envs = DMC_EASY_MEDIUM
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_hard":
        envs = DMC_HARD
        env_configs = ["dmc"] * len(envs)

    elif env_type == "HUM_STAND":
        envs = HUM_STAND
        env_configs = ["dmc"] * len(envs)
        
    elif env_type == "HUM_WALK":
        envs = HUM_WALK
        env_configs = ["dmc"] * len(envs)
        
    elif env_type == "HUM_RUN":
        envs = HUM_RUN
        env_configs = ["dmc"] * len(envs)
        
    elif env_type == "DOG_STAND":
        envs = DOG_STAND
        env_configs = ["dmc"] * len(envs)
        
    elif env_type == "DOG_WALK":
        envs = DOG_WALK
        env_configs = ["dmc"] * len(envs)
        
    elif env_type == "DOG_TROT":
        envs = DOG_TROT
        env_configs = ["dmc"] * len(envs)
        
    elif env_type == "DOG_RUN":
        envs = DOG_RUN
        env_configs = ["dmc"] * len(envs)

    elif env_type == "myosuite":
        envs = MYOSUITE_TASKS
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO1":
        envs = MYO1
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO2":
        envs = MYO2
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO3":
        envs = MYO3
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO4":
        envs = MYO4
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO5":
        envs = MYO5
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO6":
        envs = MYO6
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO7":
        envs = MYO7
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO8":
        envs = MYO8
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO9":
        envs = MYO9
        env_configs = ["myosuite"] * len(envs)
        
    elif env_type == "MYO10":
        envs = MYO10
        env_configs = ["myosuite"] * len(envs)

    elif env_type == "hb_locomotion":
        envs = HB_LOCOMOTION_NOHAND
        env_configs = ["hb_locomotion"] * len(envs)
                
    elif env_type == "HB_WALK":
        envs = HB_WALK
        env_configs = ["hb_locomotion"] * len(envs)
        
    elif env_type == "HB_CRAWL":
        envs = HB_CRAWL
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HB_STAND":
        envs = HB_STAND
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HB_RUN":
        envs = HB_RUN
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HB_HURDLE":
        envs = HB_HURDLE
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HB_MAZE":
        envs = HB_MAZE
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HB_STAIR":
        envs = HB_STAIR
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HB_POLE":
        envs = HB_POLE
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HB_SLIDE":
        envs = HB_SLIDE
        env_configs = ["hb_locomotion"] * len(envs)
        
    elif env_type == "HB_SIT1":
        envs = HB_SIT1
        env_configs = ["hb_locomotion"] * len(envs)
    
    elif env_type == "HB_SIT2":
        envs = HB_SIT2
        env_configs = ["hb_locomotion"] * len(envs)
    
    elif env_type == "HB_BALANCE1":
        envs = HB_BALANCE1
        env_configs = ["hb_locomotion"] * len(envs)
    
    elif env_type == "HB_BALANCE2":
        envs = HB_BALANCE2
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "HBH_WALK":
        envs = HBH_WALK
        env_configs = ["hb_hands"] * len(envs)
        
    elif env_type == "HBH_CRAWL":
        envs = HBH_CRAWL
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_STAND":
        envs = HBH_STAND
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_RUN":
        envs = HBH_RUN
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_HURDLE":
        envs = HBH_HURDLE
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_MAZE":
        envs = HBH_MAZE
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_STAIR":
        envs = HBH_STAIR
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_POLE":
        envs = HBH_POLE
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_SLIDE":
        envs = HBH_SLIDE
        env_configs = ["hb_hands"] * len(envs)
        
    elif env_type == "HBH_REACH":
        envs = HBH_REACH
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_SIT_H":
        envs = HBH_SIT_H
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_SIT_S":
        envs = HBH_SIT_S
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_BAL_S":
        envs = HBH_BAL_S
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_BAL_H":
        envs = HBH_BAL_H
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_BASKET":
        envs = HBH_BASKET
        env_configs = ["hb_hands"] * len(envs)
        
    elif env_type == "HBH_BOOK_H":
        envs = HBH_BOOK_H
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_BOOK_S":
        envs = HBH_BOOK_S
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_CABINET":
        envs = HBH_CABINET
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_CUBE":
        envs = HBH_CUBE
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_DOOR":
        envs = HBH_DOOR
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_INSERT_N":
        envs = HBH_INSERT_N
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_INSERT_S":
        envs = HBH_INSERT_S
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_POWERLIFT":
        envs = HBH_POWERLIFT
        env_configs = ["hb_hands"] * len(envs)
        
    elif env_type == "HBH_PUSH":
        envs = HBH_PUSH
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_ROOM":
        envs = HBH_ROOM
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_SPOON":
        envs = HBH_SPOON
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_TRUCK":
        envs = HBH_TRUCK
        env_configs = ["hb_hands"] * len(envs)

    elif env_type == "HBH_WINDOW":
        envs = HBH_WINDOW
        env_configs = ["hb_hands"] * len(envs)
        
    elif env_type == "MUJ_Cheetah":
        envs = MUJ_Cheetah
        env_configs = ["mujoco"] * len(envs)
        
    elif env_type == "MUJ_Humanoid":
        envs = MUJ_Humanoid
        env_configs = ["mujoco"] * len(envs)

    elif env_type == "MUJ_Hopper":
        envs = MUJ_Hopper
        env_configs = ["mujoco"] * len(envs)

    elif env_type == "MUJ_Walker":
        envs = MUJ_Walker
        env_configs = ["mujoco"] * len(envs)
        
    elif env_type == "MUJ_Ant":
        envs = MUJ_Ant
        env_configs = ["mujoco"] * len(envs)
        
    elif env_type == "all":
        envs = (
            DMC_EASY_MEDIUM
            + DMC_HARD
            + MYOSUITE_TASKS
            + HB_LOCOMOTION_NOHAND
        )
        env_configs = (
            ["dmc"] * len(DMC_EASY_MEDIUM)
            + ["dmc"] * len(DMC_HARD)
            + ["myosuite"] * len(MYOSUITE_TASKS)
            + ["hb_locomotion"] * len(HB_LOCOMOTION_NOHAND)
        )
        
    else:
        raise NotImplementedError

    for seed in seeds:
        for idx, env_name in enumerate(envs):
            exp = copy.deepcopy(args)  # copy overriding arguments
            exp["config_path"] = config_path
            exp["config_name"] = config_name

            exp["overrides"].append("agent=" + agent_config)
            exp["overrides"].append("env=" + env_configs[idx])
            exp["overrides"].append("env.env_name=" + env_name)

            exp["overrides"].append("server=" + server)
            exp["overrides"].append("group_name=" + group_name)
            exp["overrides"].append("exp_name=" + exp_name)
            exp["overrides"].append("seed=" + str(seed))

            experiments.append(exp)
            print(exp)

    # run parallel experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method("spawn")
    available_gpus = device_ids
    process_dict = {gpu_id: [] for gpu_id in device_ids}

    for exp in experiments:
        wait = True
        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)

            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    gpu_id, processes = min(
                        process_dict.items(), key=lambda x: len(x[1])
                    )
                    break

            time.sleep(10)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        exp["device_id"] = str(gpu_id)
        process = mp.Process(
            target=run_with_device,
            args=(
                server,
                exp["device_id"],
                exp["config_path"],
                exp["config_name"],
                exp["overrides"],
            ),
        )
        process.start()
        processes.append(process)
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)
