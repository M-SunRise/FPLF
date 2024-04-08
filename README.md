1、补充trainFPLF.py配置

    config_defaults = {
        "epochs": 30,
        "train_batch_size": 24,
        "valid_batch_size": 24,
        "gpu_nums": 1,
        "optimizer": "adam",
        "learning_rate": 2e-4,
        "weight_decay": 0,
        "rand_seed": 888,
        "accumulation_steps": 1,
    }


2、修改listener.py第15行

    cmd = 'CUDA_VISIBLE_DEVICES="GPU" python -m torch.distributed.launch --nproc_per_node GNUM --master_port 2515  trainFPLF.py >>./logs/data.log 2>&1 &'

3、视情况修改GPU数量、指定GPU编号

    NUM = 2

    GPU_list=[6,7]

4、激活环境，运行命令

    python listener.py

5、更多相关内容

    见论文或Figure文件夹

6、公开仓库 [Link](https://github.com/M-SunRise/Code_for_FPLF)