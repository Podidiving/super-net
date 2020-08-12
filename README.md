### SuperNet
Test Task for [MIL](https://github.com/machine-intelligence-laboratory/MILTestTasks/tree/task/NAS-ImageNet)

1. Related work: [Understanding and Simplifying One-Shot Architecture Search.](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf)

2. Network Description:
![SuperNet architecture](pics/super_network.png "SuperNet architecture")  


4. How to reproduce:
    You should specify corresponding `yml` file from `configs` folder for your experiment 
    1. Training
        1. To train super net from scratch: `python3 super_net/train_super_net.py -c configs/train_super_net.yml`
        2. To train specific sub network from scratch: `python3 super_net/train_sub_net.py -c configs/<choose your config here>`
    2. Evaluation
        1. Evaluate super net `python3 super_net/validate_sampled_nets.py -c configs/train_super_net.yml`
        2. Evaluate sub nets `python3 super_net/validate_sub_nets.py -c configs/<choose your config here> -p configs/model_paths.yml`
        **Note** if you changed model path in `configs/<choose your config here>` you should change corresponding model path in `configs/model_paths.yml`
