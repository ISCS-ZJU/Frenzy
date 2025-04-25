from collections import OrderedDict
from node import Node
from utils import print_fn, _repr_job_preempt, _repr_job_done, large_job_pruning
from job_history import JobHistory

class Cluster:
    def __init__(self, node_list=None, num_nodes=None, num_gpus=20,
                 num_cpus=20, pattern=1, period=124, job_list=None,
                 random_seed=0, num_spare_node=None,
                 export_cluster_util=False):
        
        # 集群节点列表
        if node_list is not None:   # 使用自定义节点列表配置
            node_list = node_list
        elif num_nodes is not None: # 使用节点默认配置：(self, id, num_gpus=8, num_cpus=96, mem=720, job_runn_list=None, gpu_type=0)
            node_list = [Node(id=i) for i in range(num_nodes)] 
        else:                       # 缺省：创建单节点集群，节点中：num_gpus=20, num_cpus=20
            node_list = [Node(id=0, num_gpus=num_gpus, num_cpus=num_cpus)]

        # 计算集群总cpu数和gpu数
        temp_node_dict = dict()
        self.num_gpus, self.num_cpus = 0, 0
        for node in node_list:
            self.num_gpus += node.num_gpus
            self.num_cpus += node.num_cpus
            temp_node_dict[node.id] = node

        # 集群节点字典，字典中元素按照节点id排好序
        self.node_dict = OrderedDict(sorted(temp_node_dict.items(),
                                            key=lambda t: t[1].id))

        self.cur_time = 0   # 集群时间
        self.svc = {'num_gpu': 0, 'num_cpu': 0} # high-priority service
                                                # 高优先级服务字典
        self.svc_former_ratio = 0

        # self.job_full_list = job_list  # all jobs received from all times
        self.job_full_list = large_job_pruning(job_list, self.num_gpus, self.num_cpus)  
                                            # 对传入的作业列表 job_list 进行修剪，以确保作业不会超出集群的 GPU 和 CPU 容量
        self.job_full_list.sort(key=lambda j: -j['submit_time'])    # 按最近提交到最早提交的顺序排序
        self.job_list = []
        self.retrieve_job_from_full_list()  # feed self.user_job_queue into self.job_list

        self.job_history = JobHistory()

        # Capacity changing pattern & period 
        # 设置集群容量变化的模式和周期。
        self.pattern = pattern
        self.period = period

        # Spare specific node
        self.num_spare_node = num_spare_node
        self.spare_node_id = []
        # 如果指定了备用节点数量，这段代码会随机选择备用节点，并将它们的 ID 添加到 self.spare_node_id 列表中。
        if num_spare_node is not None:
            for i in range(num_spare_node):
                spare_node_index = random_seed % len(node_list)
                spare_node_id = node_list[spare_node_index].id
                while spare_node_id in self.spare_node_id:
                    random_seed += 29741  # a random prime number
                    spare_node_index = random_seed % len(node_list)
                    spare_node_id = node_list[spare_node_index].id
                self.spare_node_id.append(spare_node_id) # indicate which node to spare
                random_seed += 29741  # a random prime number

        # 用于导出集群利用率的标志和相关的列表，以及一个计数器，用于记录集群空闲的时间。
        self.export_cluster_util = export_cluster_util
        self.cluster_time = []
        self.cluster_cpu = []
        self.cluster_gpu = []
        self.idle_cluster_counter = 0

        # 集群中所有GPU size类型
        self.gpu_size_list=self.get_gpu_size_list()

    # 按时间顺序处理作业列表
    # 将已经到了提交时间的作业从 self.job_full_list 移动到 self.job_list 中，以便进一步处理。
    # 如果所有剩余的作业提交时间都晚于当前时间，则停止处理并返回。
    def retrieve_job_from_full_list(self):
        while len(self.job_full_list) > 0:          # 直到所有作业提交到集群
            job = self.job_full_list[-1]
            if job['submit_time'] <= self.cur_time: # 作业提交时间 <= 集群时间，相当于作业已提交到集群
                ### add for Frenzy
                # 

                job = self.job_full_list.pop()      # 从full作业列表删除该作业
                self.job_list.append(job)           # 将进入集群的作业添加到集群作业列表
            else:
                return 0

    # 根据节点id对节点列表排序
    def sorted_node_list(self):
        node_list = list(self.node_dict.values())
        node_list.sort(key=lambda n: n.id)
        return node_list

    def tic_job(self, delta=1):
        # Unlike tic_svc(), it receives simulator's cur_time as its own cur_time
        # Here it returns a "cur_time" value to the simulator
        # If succeed: return cur_time >= 0
        # Else: return cur_time < 0 ==> exit_flag = 1
        self.cur_time += delta  # 这行代码将当前时间 self.cur_time 增加一个时间增量 delta，模拟时间步进的长度。
        if self.export_cluster_util and self.cur_time % 10000 == 0: # 是否需要导出集群利用率
            self.record_cluster_util()
        self.retrieve_job_from_full_list()  # update self.job_list
        job_runn_list = self.job_runn_list  # 正在运行的作业列表。
        if len(job_runn_list) > 0:
            for job in job_runn_list:
                # 更新作业的运行时间和进度
                job['on_time'] += delta     # job['on_time'] 表示作业已运行的时间
                job['progress'] = job['on_time'] * job['num_gpu']
                
                # Job done logic
                # 作业运行时间超过预定时间，则作业完成，释放资源，记录作业完成时间，添加历史记录
                if job['on_time'] >= job['duration']:   # 如果作业的运行时间超过了其预定持续时间
                    over_tic_time = job['on_time'] - job['duration']  # only if delta > 1
                    job['on_time'] -= over_tic_time
                    job['progress'] -= over_tic_time * job['num_gpu']
                    job['done'] = 1

                    host_node_id = job['node']
                    host_node = self.node_dict.get(host_node_id)
                    suc = host_node.release_job(job=job)
                    assert suc

                    job['jct'] = self.cur_time - over_tic_time - job['submit_time']  # deduct submit_time

                    # print(job['duration'], job['jct'])

                    self.job_history.add_done_job(job)

                    print_fn("%sDONE: %s || %s" % (self.log_prefix, _repr_job_done(job), job))
                
            return self.cur_time  # exit_flag = 0, still going

        # len(job_runn_list) <= 0,
        # 如果没有正在运行的作业，但是有等待的作业，则增加空闲集群计数器，并打印相关信息。
        elif len(self.job_list) > 0:  # empty cluster with job pending
            self.idle_cluster_counter += 1  
            print_fn("%sIDLE cluster until jobs: %s" % (self.log_prefix, [_repr_job_preempt(e) for e in self.job_list]))

            if self.idle_cluster_counter % 10000 == 0:
                print_fn('{} idle cluster: {}'.format(self.idle_cluster_counter, [_repr_job_preempt(e) for e in self.job_list]), level=2)
            return self.cur_time  # exit_flag = 0, still going

        # 如果没有正在运行的作业，也没有等待的作业，但是有即将到来的作业，则设置唤醒时间，并更新当前时间。
        elif len(self.job_full_list) > 0:  # i.e., empty cluster waiting for jobs to come   
            wake_time = self.job_full_list[-1]['submit_time'] - delta  # the submit_time of the earliest job
            assert self.cur_time <= wake_time  # if ==, i.e., the stride is unnecessary
            self.cur_time = wake_time
            return self.cur_time  # exit_flag = 0, still going
        
        # 如果没有正在运行的作业，没有等待的作业，也没有即将到来的作业，则返回 -1，表示模拟器应该退出。
        else:  # no running job, no pending job, no coming job => exit.
            return -1  # exit

    def tic_svc(self, cur_time):
        self.cur_time = cur_time
        cap_ratio = self.get_cap_ratio(cur_time)
        svc_ratio = 1 - cap_ratio
        
        # temp add, to be remove
        if svc_ratio!=0:
            print('svc_ratio:', svc_ratio)
        # temp add, to be remove

        if self.svc_former_ratio != svc_ratio:
            self.svc_former_ratio = svc_ratio
            print_fn("%sService WAS:%s" % (self.log_prefix, str([n.__repr__() for n in self.node_list])))
            for node in self.node_list:
                if node.id in self.spare_node_id:  # spare from service allocation
                    continue
                node.set_svc_res_by_ratio(ratio=svc_ratio)
            print_fn("%sService NOW:%s" % (self.log_prefix, str([n.__repr__() for n in self.node_list])))

    def replace_svc(self):
        # Migrating services or jobs for vacancies.
        raise NotImplementedError("Cluster replace service")

    def display_capacity_pattern(self, max_time=200):
        for cur_time in range(max_time):
            cur_gpus, cur_cpus = self.get_capacity(cur_time)
            four_gpus, four_cpus = int(cur_gpus / 4), int(cur_cpus / 4)
            left_gpus, left_cpus = int(cur_gpus % 4), int(cur_cpus % 4)
            print("[%3s] G%3d |%s%s\n      C%3d |%s%s" % (cur_time, cur_gpus, "####|" * four_gpus, "#" * left_gpus, cur_cpus, "xxxx|" * four_cpus, "x" * left_cpus ))
    
    def display_capacity_pattern_csv(self, max_time=200):
        print("time,GPUs,CPUs")
        for cur_time in range(max_time):
            cur_gpus, cur_cpus = self.get_capacity(cur_time)
            # four_gpus, four_cpus = int(cur_gpus / 4), int(cur_cpus / 4)
            # left_gpus, left_cpus = int(cur_gpus % 4), int(cur_cpus % 4)
            print("%d,%d,%d" % (cur_time, cur_gpus, cur_cpus))

    def get_capacity(self, time, num_spare_node=None):
        """
        Only for display_capacity_pattern()
        :param time: cluster.cur_time, cluster.num_spare_node
        :return: [cur_gpus, cur_cpus]
        """
        num_spare_node = self.num_spare_node if num_spare_node is None else num_spare_node
        ratio = self.get_cap_ratio(time)
        if num_spare_node is None:
            return [int(ratio * self.num_gpus), int(ratio * self.num_cpus)]
        else:
            if not self.spare_node_id:
                spare_node_id = list(range(num_spare_node))
            else:
                spare_node_id = self.spare_node_id
            g, c = 0, 0
            for node in self.node_list:
                if node.id in spare_node_id:
                    g += node.num_gpus
                    c += node.num_cpus
                else:
                    g += node.num_gpus - int((1 - ratio) * node.num_gpus)
                    c += node.num_cpus - int((1 - ratio) * node.num_cpus)                    
            assert g >= 0 and c >= 0
            return [g, c]

    def get_cap_ratio(self, time, pattern=None, period=None):
        pattern = self.pattern if pattern is None else pattern
        period = self.period if period is None else period

        pattern_ratio_dict = {
            0: {1:(0,1000)}, # always maximum capacity
            1: {1:(0, 62), 0.6:(62, 124)},
            2: {0.6:[(0, 10), (62, 124)], 1:(10, 62)},
            3: {1:(0, 20), 0.9:(20, 40), 0.8:(40, 60), 0.7:(60, 80), 0.6:(80, 100), 0.5:(100, 124)},
            4: {0.5:(0, 20), 0.6:(20, 40), 0.7:(40, 60), 0.8:(60, 80), 0.9:(80, 100)},
            5: {1:[(0, 10), (110, 124)], 0.9:[(10, 20),(100, 110)], 0.8:[(20, 30),(90, 100)], 0.7:[(30, 40),(80, 90)], 0.6:(40, 50), 0.5:(50, 70), 0.4:(70, 80)},
            6: {1:[(0, 20), (50, 60), (110, 124)], 0.6:(20, 50), 0.4:(60, 110)},
            7: {1:[(0, 20), (50, 60), (110, 124)], 0.9:(20, 50), 0.8:(60, 110)}
        }  # { pattern1: {ratio1: [ (lower_bound1, upper_bound1), (lb2, ub2), ... ], ratio2: [...]},  pattern2: {...}  }

        t_mod_p = time % period
        ratio_dict = pattern_ratio_dict.get(pattern, {})
        for key, val in ratio_dict.items():
            if type(val) == tuple:
                val = [val]  # becomes a list
            for bound in val:
                if bound[0] <= t_mod_p < bound[1]:
                    return key
        return 1

    def record_cluster_util(self):
        self.cluster_time.append(self.cur_time)
        self.cluster_cpu.append(self.job_cpus)
        self.cluster_gpu.append(self.job_gpus)
    
    @property
    def node_list(self):
        return list(self.node_dict.values())

    @property
    def cur_rsrc(self):
        return [self.cur_gpus, self.cur_cpus]

    @property
    def cur_gpus(self):
        return self.num_gpus - self.svc_gpus

    @property
    def cur_cpus(self):
        return self.num_cpus - self.svc_cpus

    # 将集群所有节点上正在运行的任务列表添加到集群的正在运行任务列表
    @property
    def job_runn_list(self):
        job_runn_list = []
        for node in self.node_list:
            job_runn_list.extend(node.job_runn_list)
        return job_runn_list

    @property
    def svc_gpus(self):
        return sum([n.svc_gpus for n in self.node_list])

    @property
    def svc_cpus(self):
        return sum([n.svc_cpus for n in self.node_list])

    @property
    def idl_gpus(self):
        return sum([n.idl_gpus for n in self.node_list])

    @property
    def idl_cpus(self):
        return sum([n.idl_cpus for n in self.node_list])

    @property
    def job_gpus(self):
        return sum([n.job_gpus for n in self.node_list])

    @property
    def job_cpus(self):
        return sum([n.job_cpus for n in self.node_list])

    @property
    def log_prefix(self):
        if self.export_cluster_util is True:  # add util export
            self.record_cluster_util()
        return "[%6s],[GPU,CPU]:[%7s,%8s]/[%7s,%8s]." % (self.cur_time, self.job_gpus, self.job_cpus, self.cur_gpus, self.cur_cpus)


    ###############################
    ########### Frenzy ############
    # 获取集群所有可能的GPU size列表
    def get_gpu_size_list(self):
        gpu_size_list=[]
        for node in self.node_list:
            if node.gpu_size in gpu_size_list:
                continue
            else:
                gpu_size_list.append(node.gpu_size)
        gpu_size_list.sort(reverse=False)
        return gpu_size_list
    
    # 集群资源情况
    def gpu_eq_ge(self, cluster=None):
        gpu_eq = {}         # 等于某内存的集群总gpu数, e.g {32:2, 40:3, 80:4}, key表示某个GPU内存
        gpu_ge = {}         # 大于等于某内存的集群总GPU数, e.g {32:9, 40:7, 80:4}
        free_gpu_eq = {}     # 等于某内存的可用gpu数
        free_gpu_ge = {}     # 大于等于某内存的可用gpu数

        node_list = self.node_list if cluster==None else cluster.node_list

        for node in node_list:
            gpu_size = node.gpu_size
            if gpu_size not in gpu_eq:
                gpu_eq[gpu_size] = node.num_gpus / 100    # not in %
            else:
                gpu_eq[gpu_size] += node.num_gpus / 100

            if gpu_size not in free_gpu_eq:
                free_gpu_eq[gpu_size] = node.idl_gpus / 100  # not in %
            else:
                free_gpu_eq[gpu_size] += node.idl_gpus / 100  

        # 对gpu_eq和free_gpu_eq按照key(GPU大小进行降序排序)
        gpu_eq = dict(sorted(gpu_eq.items(), key=lambda item: item[0], reverse=True))
        free_gpu_eq = dict(sorted(free_gpu_eq.items(), key=lambda item: item[0], reverse=True))

        gpu_size_list = list(gpu_eq.keys()) # 所有可能的gpu size列表, 已经按照size大小降序排序

        # 获得gpu_ge与free_gpu_ge
        for i in range(len(gpu_size_list)):
            gpu_size = gpu_size_list[i]
            if i==0:
                gpu_ge[gpu_size] = gpu_eq[gpu_size]
                free_gpu_ge[gpu_size] = free_gpu_eq[gpu_size]
            else:
                gpu_size_last = gpu_size_list[i-1]
                gpu_ge[gpu_size] = gpu_ge[gpu_size_last] + gpu_eq[gpu_size]
                free_gpu_ge[gpu_size] = free_gpu_ge[gpu_size_last] + free_gpu_eq[gpu_size]       

        return gpu_eq, gpu_ge, free_gpu_eq, free_gpu_ge