import utils
from utils import print_fn, ALLOC_POLICY_DICT, PREEMPT_POLICY_DICT, _repr_job_concise

import math
from Frenzy import MARP, Scheduler as scheduler_helper, Duration

class Scheduler:
    def __init__(self, alloc_policy=0, preempt_policy=0, sort_node_policy=0, cluster=None, gpu_type_matching=0, gpu_size_matching=0, verbose=0):
        self.cluster = cluster
        self.alloc_policy = alloc_policy
        self.preempt_policy = preempt_policy
        self.sort_node_policy = sort_node_policy
        self.node_rotate_counter = 0
        self.verbose = verbose
        self.gpu_type_matching = gpu_type_matching
        self.gpu_size_matching = gpu_size_matching
        # To skip unnecessary self.alloc_job_sort()
        self.last_time_snapshot = [0, 0, 0, 0]  # [idle_gpu, idle_cpu, len(job_list), len(job_to_allocate_cache)]
        self.cannot_counter = 0

    def alloc_job(self, cluster=None):
        cluster = cluster if cluster is not None else self.cluster
        job_list = cluster.job_list  # Take cluster.job_list

        # Trying skipping allocation as early as possible
        if len(job_list) <= 0:
            return 0
        ig, ic = cluster.idl_gpus, cluster.idl_cpus
        this_time_snapshot = [ig, ic, len(job_list), 0]  # 0: no job allocated.
        if self.last_time_snapshot == this_time_snapshot:  # exactly the same
            if self.verbose:
                print_fn("[{}] Last time snapshot == this time snapshot: {}. Bypass.".format(self.cluster.cur_time, this_time_snapshot))
            return 0
        job_min_gpu, job_min_cpu = min(job_list, key=lambda j: j['num_inst'] * j['num_gpu']), min(job_list, key=lambda j: j['num_inst'] * j['num_cpu'])
        if (ig <= 0 or job_min_gpu['num_inst'] * job_min_gpu['num_gpu'] > ig) and (ic <= 0 or job_min_cpu['num_inst'] * job_min_cpu['num_cpu'] > ic):
            self.last_time_snapshot = this_time_snapshot
            return 0

        if self.verbose:
            print_fn("job_min_gpu, job_min_cpu = {:.1f}, {:.1f}".format(job_min_gpu['num_gpu'], job_min_cpu['num_cpu']))

        job_to_allocate_cache = []
        # Greedy algorithm or Greedy + load balancing
        if self.alloc_policy in ALLOC_POLICY_DICT.keys():
            # Heavy action
            self.alloc_job_sort(job_list, cluster.job_runn_list)
            for job_a in job_list:
                succ_alloc = self.try_allocate_job_to_cluster(job_a, cluster)
                if succ_alloc == 1:
                    job_to_allocate_cache.append(job_a)
                elif succ_alloc == -1:
                    break
                # else, e.g., succ_alloc == 0: pass/continue
        else:
            raise KeyError("Uncaptured Allocation Policy Input: %d" % self.alloc_policy)

        this_time_snapshot[-1] = len(job_to_allocate_cache)  # num of jobs allocated
        self.last_time_snapshot = this_time_snapshot
        for job_a in job_to_allocate_cache:
            cluster.job_list.remove(job_a)

    def alloc_job_sort(self, job_list, job_runn_list=None):
        if self.alloc_policy == 0:  # short_duration_first
            job_list.sort(key=lambda e: (e['duration'], e['job_id']))
        elif self.alloc_policy == 8:  # FIFO, remains the original order
            job_list.sort(key=lambda e: (e['submit_time'], e['job_id']))
        elif self.alloc_policy in [1, 2, 4]:  # SJF with duration estimation
            est_feature = {1: 'user_dur', 2: 'group_dur', 4: 'group_gpu_dur'}[self.alloc_policy]
            job_list.sort(key=lambda e: (e[est_feature], e['job_id']))
        else:
            raise Exception("Unexpected alloc policy: %d" % self.alloc_policy)

        if self.verbose:
            for i, j in enumerate(job_list):
                print_fn("%2d %s" % (i, j))
                if i > 20:
                    break
 
    def try_allocate_job_to_cluster(self, job_a, cluster):
        """
        job_a: job to allocate
        cluster: target cluster

        return:
            -1: the cluster is full, stop job picking
             0: the current job cannot be placed, try next
             1: the current job has been successfully deployed, need record.
        """

        ''' PART1 获得合适的配置
            1. 设置模型超参数和训练参数
            2. 利用MARP计算得到一系列资源配置, 我们需要根据集群的资源情况选择其中一条合适的配置
               这里认为MARP返回的一些列推荐配置中, 第一条的效果最好, 往后依效果次减弱
            3. 获得基于集群目前资源情况的最优资源配置
               该配置只是针对该单一作业的、在集群当前可用资源情况下的最优解
        '''
        # model configuration hyperparameters
        modelCfg = dict(
            layers=job_a['layer'],
            atten_heads=job_a['atten_head'],
            hidden_dimension=job_a['hid_dim'],
            seq_length=job_a['seq_len'],
            vocab_size=job_a['voca_size'],
        )

        # model training parameters
        trainCfg = dict(
            global_batch_size=job_a['g_batch_size'],
        )

        # get possible resource allocation schemes (rsc_cfg)
        rsc_cfgs = MARP.MARP(modelCfg, trainCfg)
        
        # get a specific resources configuration
        gpu_eq, gpu_ge, free_gpu_eq, free_gpu_ge = cluster.gpu_eq_ge(cluster)
        # 资源调度
        rsc_cfg = None
        for cfg in rsc_cfgs:
            mem = cfg['mem_per_gpu']
            gpu_num = cfg['gpu_nums']
            size = scheduler_helper.gpu_ceil(mem, cluster.gpu_size_list)
            cfg['gpu_size'] = size  # 更新cfg['gpu_size']

            if size==-1:
                continue
            else:
                if gpu_num <= free_gpu_ge[size]: # 能够分配
                    # gpu_alloc_list = scheduler_helper.get_gpu_alloc_list(size, gpu_num, cluster.gpu_size_list, free_gpu_eq) # 基于集群可用资源获得合适的资源分配方案
                    # node_gpu_used = scheduler_helper.allocate_gpu(gpu_alloc_list, cluster)   # 基于资源分配方案进行资源分配, 获得每个节点的资源占用情况
                    rsc_cfg = cfg
                # else:
                #     continue
        
        if rsc_cfg == None:
            if scheduler_helper.job_can_alloc_to_cluster(rsc_cfgs, cluster.gpu_size_list, gpu_ge) == -1:
                raise RuntimeError("Even the entire cluster is unable to meet the requirements for this job.")
            else:
                return 0 # the current job cannot be placed, try next

        trainCfg['DP'] = rsc_cfg['dp']
        trainCfg['TP'] = rsc_cfg['tp']
        trainCfg['train_steps'] = job_a['train_steps']

        job_a['num_inst']=trainCfg['DP']*trainCfg['TP']
        job_a['num_gpu'] = 100 # in %
        job_a['num_cpu'] = 0
        job_a['gpu_size']=rsc_cfg['gpu_size']

        ig, ic = cluster.idl_gpus, cluster.idl_cpus
        if ig <= 0 and ic <= 0:
            return -1
        elif job_a['num_inst'] * job_a['num_gpu'] > ig or job_a['num_inst'] * job_a['num_cpu'] > ic:
            return 0
        else:  # with in gpu and cpu limits
            assigned_node_map = {}
            assigned_inst_num = 0
            
            ''' PART2 为作业进行资源调度, 基本思路为: 
                1. 将节点列表按照空闲资源升序、FLOPs降序进行排序 即sorted_node_list = self.sorted_node_list(cluster.node_list)
                2. 对一个job_a, 前面已经计算好了需要的GPU数, 记为req_num_gpu (初始时为job_a['num_inst'])
                3. 理想情况是正好有一个节点有job_a['num_inst']块空闲的GPU, 此时直接将该节点分配给该job_a
                4. 如果没有能够正好满足的节点, 但是有能满足job_a的节点, 选择能满足作业的节点中剩余GPU最少的节点
                   如: ABCD三个节点分别有2,3,5,6块GPU, job需要6块GPU, 则选择C节点
                5. 如果没有单个节点能满足job_a, 则按照如下方案进行调度:
                   将所剩资源最大的节点分配给该任务, 然后更新req_num_gpu
                   基于最新的req_num_gpu, 继续执行3-5步, 直到满足job_a的资源需求, 或者任务分配失败。

                其中, 调度时考虑了GPU size, 先遍历严格满足job_a['gpu_size']的节点, 如果不足以满足需求, 继续遍历更大gpu_size的节点
                如节点A:16GB*4, B:24GB*6, C:32GB*2, D:40GB*4, E:80GB*2, 对一个所需资源为24GB*10的job: 
                    调度时先严格限制只能用24GB的节点, 24GB的节点全部分配给该job;
                    发现不够, 再严格限制只能用32GB的节点, 32GB的节点全部分配给该job;
                    发现不够, 在严格限制只能用40GB的节点, 足以满足要求, 完成分配。             
            '''
            sorted_node_list = self.sorted_node_list(cluster.node_list)

            # 对剩下(包括最开始)需要分配的inst选择最合适的节点
            gpu_size_list=cluster.get_gpu_size_list()
            gpu_size_list.sort()
            size_idx = gpu_size_list.index(job_a['gpu_size'])
            match_size = gpu_size_list[size_idx]
            last_use_id = len(sorted_node_list)
            while assigned_inst_num < job_a['num_inst']:
                one_node_enough = False     # 是否存在一个节点直接满足剩下所有的需求
                max_node = None
                max_id = -1
                # last_use_nid = len(sorted_node_list)
                # node按idl_gpus从小到大排序
                for id, node in enumerate(sorted_node_list):
                    if id>=last_use_id:
                        break
                    # <Node-job label matching>
                    if self.gpu_type_matching == 1:     # GPU type perfect match. 
                        if job_a['gpu_type'] != 'CPU' and job_a['gpu_type'] != node.gpu_type:
                            continue  # cannot on this node
                    elif self.gpu_type_matching == 2:  # Only V100 cannot compromise
                        if job_a['gpu_type'] == 'V100' and job_a['gpu_type'] != node.gpu_type:
                            continue  # cannot on this node

                    if self.gpu_size_matching == 1:     # GPU size perfect match. 
                        if node.gpu_size != match_size:
                            continue  # cannot on this node
                    # </Node-job label matching>

                    max_node = node
                    max_id = id
                    
                    # 该节点剩余资源
                    node_idle_gpus, node_idle_cpus = node.idl_gpus, node.idl_cpus
                    node_inst_num_gpu, node_inst_num_cpu = job_a['num_inst'], job_a['num_inst']  # init.
                    if job_a['num_gpu'] != 0:
                        node_inst_num_gpu = node_idle_gpus // job_a['num_gpu']
                    if job_a['num_cpu'] != 0:
                        node_inst_num_cpu = node_idle_cpus // job_a['num_cpu']
                    # 该节点可提供的inst数
                    node_inst_num = min(node_inst_num_gpu, node_inst_num_cpu)

                    # 如果该节点能够满足剩下所有资源需求，使用该节点
                    if assigned_inst_num + node_inst_num >= job_a['num_inst']:
                        node_inst_num = job_a['num_inst'] - assigned_inst_num   # 还需要的inst数，该节点能够满足
                        assigned_node_map[node.id] = node_inst_num
                        assigned_inst_num += node_inst_num
                        one_node_enough = True
                        break
                    # 该节点无法满足所需需求，探究下一个节点
                    else:
                        continue

                # 如果最大的节点也无法满足需求，直接使用最大节点
                if not one_node_enough:
                    if max_node is not None:
                        node = max_node
                        last_use_id=max_id
                        # 该节点剩余资源
                        node_idle_gpus, node_idle_cpus = node.idl_gpus, node.idl_cpus
                        # print(node.id, ': ', node_idle_gpus)
                        node_inst_num_gpu, node_inst_num_cpu = job_a['num_inst'], job_a['num_inst']  # init.
                        if job_a['num_gpu'] != 0:
                            node_inst_num_gpu = node_idle_gpus // job_a['num_gpu']
                        if job_a['num_cpu'] != 0:
                            node_inst_num_cpu = node_idle_cpus // job_a['num_cpu']
                        # 该节点可提供的inst数
                        node_inst_num = min(node_inst_num_gpu, node_inst_num_cpu)
                        if node_inst_num > 0:
                            assigned_node_map[node.id] = node_inst_num
                            assigned_inst_num += node_inst_num
                    else:
                        if size_idx<len(gpu_size_list)-1:
                            size_idx+=1
                            match_size=gpu_size_list[size_idx]
                            last_use_id = len(sorted_node_list)
                        else:
                            return 0    # 当前作业无法满足

                else:
                    break

            # print('job_inst:', job_a['num_inst'], 'gpu_size:', job_a['gpu_size'],  'node_assig:', assigned_node_map)
            if assigned_inst_num < job_a['num_inst']:
                print_fn("Cannot allocate all instances (%d/%d) of %s." % (assigned_inst_num, job_a['num_inst'], _repr_job_concise(job_a)))
                self.cannot_counter += 1
                if self.cannot_counter % 100000 == 0:
                    print_fn("[%s] %d rejects. len(job_done_list) = %d. Current job: %s." % (cluster.log_prefix, self.cannot_counter, len(self.cluster.job_history.job_done_list), _repr_job_concise(job_a)))
                return 0  # No successful allocation, for num_inst=1 and >1 cases
            else:
                # Successfully Scheduled. Assigning instances to nodes according to the map
                inst_id = 0

                # calculate duration #################
                min_FLOPs = 999999*(10**12)
                for nid, _ in assigned_node_map.items():
                    node = cluster.node_list[nid]
                    min_FLOPs = min(min_FLOPs, node.gpu_FLOPs)
                trainCfg['FLOPS'] = min_FLOPs    # select the minimum FLOPS among all allocated nodes
                duration = Duration.get_train_duration(modelCfg, trainCfg)
                job_a['duration']=math.floor(duration)
                # calculate duration #################

                for nid, num_inst in assigned_node_map.items():
                    node = cluster.node_list[nid]
                    job_tmp = {'node': -1}
                    for _ in range(num_inst):
                        job_tmp = job_a.copy()
                        job_tmp['inst_id'] = inst_id
                        succ_alloc = node.alloc_job(job_tmp)
                        assert succ_alloc
                        job_tmp['node'] = node.id
                        print_fn("%sON  : N[%d] %s Inst[%d]" % (cluster.log_prefix, job_tmp['node'], job_tmp, inst_id))
                        inst_id += 1
                    self.display_node_status(cur_node_id=job_tmp['node'])
                assert inst_id == job_a['num_inst']
                return 1

    def sorted_node_list(self, node_list, req_sum_gpu=None): # actually equal to num_inst * num_gpus
        policy = self.sort_node_policy
        if policy == 0:
            node_list.sort(key=lambda n: n.id)  # by id
        elif policy == 1:
            node_list.sort(key=lambda n: n.idl_gpus)    # smallest idle gpus first
        elif policy == 2:
            node_list.sort(key=lambda n: -n.idl_gpus)   # largest idle gpus first
        elif policy == 3:
            node_list.sort(key=lambda n: n.util_rate)   # lowest avg. util. first 
        elif policy == 4:
            node_list.sort(key=lambda n: (n.idl_gpus, -n.gpu_FLOPs)) # 根据可用GPU和FLOPs排序，都按从大到小的顺序排序
        else:
            node_list.sort(key=lambda n: n.id)
        return node_list

    def preempt_job(self, cluster=None):
        cluster = cluster if cluster is not None else self.cluster
        if all([n.idl_gpus for n in cluster.node_list]) >= 0 and \
            all([n.idl_cpus for n in cluster.node_list]) >= 0:
            return 0  # No resource contention, bypass preemption

        preempted_job_list = []
        if self.preempt_policy in PREEMPT_POLICY_DICT.keys():
            # Pre node preemption: self.preempt_job_node(node)
            for node in cluster.node_list:
                # As long as the resources are sufficient, no proactive preempt for now.
                if node.idl_gpus < 0 or node.idl_cpus < 0 or len(preempted_job_list) > 0:
                    print_fn("%sPreempt jobs on %s" % (cluster.log_prefix, node))
                    preempted_job_list = self.preempt_job_node(node, preempted_job_list)
            for job in preempted_job_list:
                print_fn("%sOFF : %s" % (cluster.log_prefix, job))
        else:
            raise NotImplementedError("Preempting job policies not implemented")

        for job in preempted_job_list:
            cluster.job_list.append(job)
            # Update Job
            job['wasted'] += job['progress']
            job['on_time'] = 0
            job['progress'] = 0
            job['node'] = None

    def preempt_job_node(self, node, preempted_job_list):
        # Svc is updated, but the job is not
        node.update_idl_gpus()
        node.update_idl_cpus()

        if self.preempt_policy in PREEMPT_POLICY_DICT.keys():
            # Sort node.job_runn_list in place
            self.preempt_job_sort_node(node=node, preempt_policy=self.preempt_policy)

            for job_i in preempted_job_list:
                for job_j in node.job_runn_list:
                    if job_i['job_id'] == job_j['job_id']:  # these instances belong to the same job
                        succ = node.release_job(job_i)
                        assert succ is True
                        preempted_job_list.append(job_i)

            while node.idl_gpus < 0 or node.idl_cpus < 0:
                job_to_preempt = node.job_runn_list[0]
                succ = node.release_job(job_to_preempt)
                assert succ is True
                preempted_job_list.append(job_to_preempt)

        else:
            raise KeyError("Uncaptured Preemption Policy Input: %d" % self.preempt_policy)

        return preempted_job_list

    def preempt_job_sort_node(self, node, preempt_policy):
        if preempt_policy == 1: # small_size_first
            node.job_runn_list.sort(key=lambda e: (e['size'], e['job_id']))
        elif preempt_policy == 2: # large_gang_first
            node.job_runn_list.sort(key=lambda e: (-e['num_gpu'], e['job_id']))
        else: # preempt_policy==0 or others: short_duration_first
            node.job_runn_list.sort(key=lambda e: (e['duration'], e['job_id']))

    def display_node_status(self, cur_node_id):
        if cur_node_id >= 0:
            cur_node = self.cluster.node_dict[cur_node_id]
            print_fn(cur_node)
