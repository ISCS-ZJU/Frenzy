'''
Memory Aware Resources Predictor

'''

def MARP(ModelConfig, TrainConfig, tp_range=range(1, 65), dp_range=[2**i for i in range(0, 10)], min_gpu_size=20):
    '''
    根据模型超参数配置计算所需内存，给出可能的解决方案
    将给出多种解决方案，以更适应集群的资源情况
    资源调度器选择最合适的资源配置

    Args:
        ModelConfig(dict): 模型配置(超参数)
        TrainConfig(dict): 训练配置
        tp_range(list): 模型并行可能的范围
        dp_range(list): 数据并行可能的范围
        min_gpu_size(int): 各种GPU中最小的GPU内存大小(GB)

    Returns:
        rsg_cfg(list: 可能的资源配置列表, 按所需GPU数从小到大排列
    '''
    # 模型参数
    mc_layerNum = ModelConfig['layers'] 
    mc_attenHeads = ModelConfig['atten_heads']
    mc_hidDim = ModelConfig['hidden_dimension']
    mc_seqLength = ModelConfig['seq_length']

    mc_vocabSize = ModelConfig['vocab_size']

    # 训练参数
    tc_gBatchSize = TrainConfig['global_batch_size']

    # 计算中间量
    ## 模型参数量
    paras_num =  12 * mc_layerNum * mc_hidDim * mc_hidDim + 13 * mc_layerNum * mc_hidDim + mc_vocabSize * mc_hidDim
    print(f'模型参数量: {paras_num/(10**6):.1f}M ({paras_num})')

    # 维护一个可能的资源配置
    # 每一条记录包含，每块GPU所需内存、模型并行数、数据并行数、所需gpu数
    resource_config = []
    
    # Tensor Parallel (Model Parallel)
    for tp in tp_range:
        # 张量并行是以注意力头划分的
        if mc_attenHeads % tp != 0:
            continue

        # 模型参数占用内存(bytes)
        paras_mem = (2+4) * paras_num / tp
        # 优化器状态占用内存(bytes) 
        os_mem = (4+4) * paras_num / tp
        # 梯度占用内存(bytes)
        grad_mem = (2+4) * paras_num / tp

        # Data Parallel
        for dp in dp_range:
            if tc_gBatchSize // dp == 0:
                break

            # 激活占用内存(bytes)
            acti_mem = tc_gBatchSize / dp * mc_seqLength * mc_hidDim * mc_layerNum \
                   * (10 + 24/tp + 5 * mc_attenHeads * mc_seqLength / (mc_hidDim * tp )) 
            # print(f'激活：{acti_mem}')
            
            # 每块GPU需要的内存(GB)
            #mem_per_gpu = (paras_mem + os_mem + grad_mem + acti_mem)/(2**30)
            mem_per_gpu = (paras_mem + os_mem + grad_mem + acti_mem)/(10**9)
            # mem_per_gpu = (paras_mem + os_mem + grad_mem + acti_mem)

            rsc_cfg_item = {'mem_per_gpu':mem_per_gpu, 'gpu_nums':tp*dp, 'tp':tp, 'dp':dp}
            
            resource_config.append(rsc_cfg_item)

            # if mem_per_gpu < min_gpu_size * 0.95:
            #     break

    # 对resource_config进行排序，先对其按gpu_nums升序排序；对gpu_nums相同的，再按mem_per_gpu升序排序
    # 贪心策略：所用gpu数最少，每张gpu所用内存最少
    resource_config = sorted(resource_config, key=lambda x: (x['gpu_nums'], x['mem_per_gpu']))
    return resource_config

def main():
    # A test for GPT3
    # ModelConfig = {'layers': 96, 'atten_heads': 96, 'hidden_dimension': 12288, 'seq_length': 2048, 'vocab_size':0}
    # TrainConfig = {'global_batch_size': 64}

    # GPT2
    # ModelConfig = {'layers': 12, 'atten_heads': 12, 'hidden_dimension': 768, 'seq_length': 512, 'vocab_size':0}
    # TrainConfig = {'global_batch_size': 8}

    # BERT
    # ModelConfig = {'layers': 12, 'atten_heads': 12, 'hidden_dimension': 768, 'seq_length': 512, 'vocab_size':28996}
    # TrainConfig = {'global_batch_size': 8}

    #ModelConfig = {'layers':24, 'atten_heads': 16, 'hidden_dimension': 1024, 'seq_length': 1024, 'vocab_size':28996}
    #TrainConfig = {'global_batch_size': 4}

    ModelConfig = {'layers':32, 'atten_heads': 32, 'hidden_dimension': 4096, 'seq_length': 2048, 'vocab_size':28996}
    TrainConfig = {'global_batch_size': 1}
    
    #ModelConfig = {'layers':40, 'atten_heads': 40, 'hidden_dimension': 5120, 'seq_length': 2048, 'vocab_size':28996}
    #TrainConfig = {'global_batch_size': 1}

    #ModelConfig = {'layers':2, 'atten_heads': 4, 'hidden_dimension': 64, 'seq_length': 64, 'vocab_size':28996}
    #TrainConfig = {'global_batch_size': 1}

    #ModelConfig = {'layers':16, 'atten_heads': 16, 'hidden_dimension': 512, 'seq_length': 1024, 'vocab_size':28996}
    #TrainConfig = {'global_batch_size': 1}
    
    rsc_cfg = MARP(ModelConfig, TrainConfig)
    for i in range(len(rsc_cfg)):
        print(rsc_cfg[i])
    
if __name__ == '__main__':
    main()