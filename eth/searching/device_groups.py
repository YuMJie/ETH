from typing import List
import itertools
import json
from itertools import chain

from .state import SearchStage

MAX_VALUE = 2**30


class ListElement:
    def __init__(self, value, next):
        self.value = value
        self.next = next
    def nth(self, n):
        o = self
        i = 0
        while i < n and o.next is not None:
            o = o.next
            i += 1
        return o

def init(multiset):
    multiset.sort() # ensures proper non-increasing order
    h = ListElement(multiset[0], None)
    for item in multiset[1:]:
        h = ListElement(item, h)
    return h, h.nth(len(multiset) - 2), h.nth(len(multiset) - 1)

def visit(h):
    """Converts our bespoke linked list to a python list."""
    o = h
    l = []
    while o is not None:
        l.append(o.value)
        o = o.next
    return l

def permutations(multiset):
    """Generator providing all multiset permutations of a multiset."""
    h, i, j = init(multiset)
    yield visit(h)
    while j.next is not None or j.value < h.value:
        if j.next is not None and i.value >= j.next.value:
            s = j
        else:
            s = i
        t = s.next
        s.next = t.next
        t.next = h
        if t.value < h.value:
            i = t
        j = i.next
        h = t
        yield visit(h)












from itertools import combinations_with_replacement, permutations


import itertools

def enumerate_homogeneous_stage_sizes(n, m, min_value, max_value):
    possible_numbers = []
    x = 1  # 2^0
    while x <= max_value:
        if x >= min_value:
            possible_numbers.append(x)
        x *= 2

    if not possible_numbers or n <= 0:
        return []

    all_combos = itertools.product(possible_numbers, repeat=n)
    # print(all_combos)
    result = [list(combo) for combo in all_combos if sum(combo) == m]

    return result



import itertools

def enumerate_node_aligned_stage_sizes(n, m, min_value, max_value, clist):
    # 检查clist的总和是否等于m
    if sum(clist) != m:
        return []

    # 生成所有可能的2的幂次方数，且在[min_value, max_value]范围内
    possible_numbers = []
    x = 1  # 2^0
    while x <= max_value:
        if x >= min_value:
            possible_numbers.append(x)
        x *= 2

    # 处理边界情况
    if not possible_numbers or n <= 0 or len(clist) == 0:
        return []

    # 生成所有可能的n元组，并过滤总和为m的组合
    all_combos = itertools.product(possible_numbers, repeat=n)
    filtered_combos = (combo for combo in all_combos if sum(combo) == m)

    # 检查每个组合是否符合clist的分割条件
    valid_combos = []
    for combo in filtered_combos:
        target_index = 0
        current_sum = 0
        valid = True
        for num in combo:
            if target_index >= len(clist):
                valid = False
                break
            current_sum += num
            if current_sum == clist[target_index]:
                target_index += 1
                current_sum = 0
            elif current_sum > clist[target_index]:
                valid = False
                break
        # 所有元素处理完毕后，必须所有目标都完成且current_sum为0
        if valid and target_index == len(clist) and current_sum == 0:
            valid_combos.append(list(combo))

    return valid_combos




def get_memory_limit(args,node_id_num):
    min_memory = MAX_VALUE
    for node_id, num in node_id_num.items(): #TODO can be optimized
        min_memory = min(min_memory, args.device_info[node_id]["memory"])
    # print(f"min_memory: {min_memory}")
    return min_memory

def check_memory_legacy(args,config):
    assert len(config.memory_list) == len(config.stages) , f"memory list length not match, which {len(config.memory_list)} != {len(config.stages)}"

    for i in range(config.num_stages):
        min_memory = get_memory_limit(args,config.stages[i].node_id_num)
        # print(f"stage {i}, config.memory_list[i] {config.memory_list[i]}, min_memory: {min_memory}")
        if min_memory < config.memory_list[i]:
            return False
    return True

def get_max_tp(args,stage:SearchStage,node_id_num =None): #TODO when tensor parallelism can be  split different workloads to different nodes
    max_tp_group = MAX_VALUE

    if(node_id_num is not None):
        for node_id, num in node_id_num.items():
            max_tp_group = min(max_tp_group,num)

        return max_tp_group

    for node_id, num in stage.node_id_num.items():
        max_tp_group = min(max_tp_group,num)
    return min(max_tp_group,args.max_tp)


def gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list): # gpu_num_list = [1,2,4,8] node_id_num = {0:{A100:1},1:{A100:2},2:{A100:4},3:{A100:8}}
    num_gpus_per_node_list = args.num_gpus_per_node_list
    gpu_type_list = args.gpu_type_list
    node_id_num = {}
    current_node_id = 0
    current_gpu_id = 0
    # print(gpu_type_list,num_gpus_list,num_gpus_per_node_list)
    for i in range(len(num_gpus_list)):
        num_gpus = num_gpus_list[i]
        node_id_num[i] = {}
        while num_gpus > 0:
            if(gpu_type_list[current_node_id] not in node_id_num[i].keys()):
                node_id_num[i][gpu_type_list[current_node_id]] = 0
            add = min(num_gpus, num_gpus_per_node_list[current_node_id]-current_gpu_id)
            node_id_num[i][gpu_type_list[current_node_id]] += add
            num_gpus -= add
            current_gpu_id += add
            if current_gpu_id == num_gpus_per_node_list[current_node_id]:
                current_gpu_id = 0
                current_node_id += 1
    return node_id_num




class BipartiteMatcher:
    def __init__(self, a_list, b_list):
        self.a_list = a_list
        self.b_list = b_list
        self.adj = [[] for _ in range(len(a_list))]
        for a_idx, a_gpu in enumerate(a_list):
            for b_idx, b_gpu in enumerate(b_list):
                if b_gpu[0] >= a_gpu[0] and b_gpu[1] >= a_gpu[1]:
                    self.adj[a_idx].append(b_idx)
        self.match_to = [-1] * len(b_list)

    def find_max_matching(self):
        result = 0
        for a in range(len(self.a_list)):
            visited = [False] * len(self.b_list)
            if self.dfs(a, visited):
                result += 1
        return result

    def dfs(self, a, visited):
        for b in self.adj[a]:
            if not visited[b]:
                visited[b] = True
                if self.match_to[b] == -1 or self.dfs(self.match_to[b], visited):
                    self.match_to[b] = a
                    return True
        return False

def solution(A, B):
    a_gpus = [(gpu['tflops'], gpu['memory']) for gpu in A.values()]
    b_gpus = [(gpu['tflops'], gpu['memory']) for gpu in B.values()]

    if len(b_gpus) < len(a_gpus):
        return False

    matcher = BipartiteMatcher(a_gpus, b_gpus)
    max_match = matcher.find_max_matching()

    return max_match == len(a_gpus)


def get_upper_bound(pipeline_split,cached_throughput,prefix2dict,pipeline_split_start,args):

    proven_upper_bound = [0 ] * pipeline_split_start
    approximate_upper_bound = [0 ] * pipeline_split_start

    def get_tflops_rate(A_gpu_set,B_gpu_set):
        memory_A = 0
        memory_B = 0
        tflops_A =0
        tflops_B =0
        for key in A_gpu_set.keys():
            memory_A+=A_gpu_set[key]["memory"]
            tflops_A+=A_gpu_set[key]["tflops"]
        for key in B_gpu_set.keys():
            memory_B+=B_gpu_set[key]["memory"]
            tflops_B+=B_gpu_set[key]["tflops"]

        return max(memory_A*tflops_A/memory_B/tflops_B,1)

    A_gpu_set = {}
    for pipe_idx in range(pipeline_split_start,len(pipeline_split)): # 每一个pipeline_split[pipe_idx] 是一条pipeline
        proven_upper_bound.append(10000)
        approximate_upper_bound.append(10000)
        gpu_type_num_dict_ = pipeline_split[pipe_idx]
        gpu_type_num_dict = {}
        idx = 0
        for item in gpu_type_num_dict_:
            gpu_type_num_dict[f"{idx}"] = item
            idx += 1
        for idx in gpu_type_num_dict.keys():
            A_gpu_set[idx] = gpu_type_num_dict[idx]
            A_gpu_set[idx]["memory"] = args.device_info[A_gpu_set[idx]["GPU"]]["memory"]
            A_gpu_set[idx]["tflops"] = args.device_info[A_gpu_set[idx]["GPU"]]["TFLOPS"]

        for key in cached_throughput.keys():
            B_gpu_set = {}

            for idx in prefix2dict[key].keys():
                B_gpu_set[idx] = prefix2dict[key][idx]
                B_gpu_set[idx]["memory"] = args.device_info[B_gpu_set[idx]["GPU"]]["memory"]
                B_gpu_set[idx]["tflops"] = args.device_info[B_gpu_set[idx]["GPU"]]["TFLOPS"]

            approximate_upper_bound[pipe_idx] = min(args.upper_bound_rate * get_tflops_rate(A_gpu_set, B_gpu_set) * cached_throughput[key], approximate_upper_bound[pipe_idx])
            if(solution(A_gpu_set,B_gpu_set)):
                # print("A_gpu_set",A_gpu_set ,"gpu_type_num_dict",gpu_type_num_dict, "B_gpu_set",B_gpu_set)
                proven_upper_bound[pipe_idx] = min(cached_throughput[key], proven_upper_bound[pipe_idx])

    result = [ min(proven_upper_bound[pipe_idx] , approximate_upper_bound[pipe_idx]) for pipe_idx in range(len(pipeline_split))]
    return result, proven_upper_bound,approximate_upper_bound


# A =  {'0': { 'tflops': 8 , 'memory': 10 }, '1': { 'tflops': 18 , 'memory': 20}, '2': { 'tflops': 18 , 'memory': 20},'3': { 'tflops': 8 , 'memory': 10 } }
# B = {'0': { 'tflops': 18 , 'memory': 20 }, '1': {'tflops': 9 , 'memory': 10}, '2': { 'tflops': 1 , 'memory': 1}    }
import numpy as np
def sort_pipeline_split_list(pipeline_split_list,args):
    if(not args.sort_pipeline):
        return pipeline_split_list
    loss_list = []
    var_inter_gpu_list = []
    var_intra_gpu_list = []
    for pipeline_split in pipeline_split_list:
        inter_gpu = []
        var_intra_gpu = 0
        for pipe_idx in range(len(pipeline_split)): # 每一个pipeline_split[pipe_idx] 是一条pipeline

            intra_gpu = []
            gpu_type_num_dict_ = pipeline_split[pipe_idx]
            gpu_type_num_dict = {}
            idx = 0
            for item in gpu_type_num_dict_:
                gpu_type_num_dict[f"{idx}"] = item
                idx += 1
            for idx in gpu_type_num_dict.keys():
                for num_gpu in range(gpu_type_num_dict[idx]["GPU_NUM"]):
                    intra_gpu.append(args.device_info[gpu_type_num_dict[idx]["GPU"]]["TFLOPS"] )
            # print("intra_gpu",intra_gpu)
            inter_gpu.append(sum(intra_gpu))
            var_intra_gpu += np.var(intra_gpu)
        var_inter_gpu = np.var(inter_gpu)
        loss = var_intra_gpu/args.num_layers + var_inter_gpu * len(pipeline_split) / (args.global_batch_size)
        loss_list.append(loss)
        var_inter_gpu_list.append(var_inter_gpu * len(pipeline_split) / (args.global_batch_size))
        var_intra_gpu_list.append(var_intra_gpu/args.num_layers)
        # print("pipeline_split",pipeline_split)
        # print("loss",loss)
        print("var_inter_gpu", var_inter_gpu)
        print("var_intra_gpu", var_intra_gpu)
    sorted_pipeline_split_list = [x for _, x in sorted(zip(loss_list, pipeline_split_list))]
    sorted_var_intra_gpu_list = [x for _, x in sorted(zip(loss_list, var_intra_gpu_list))]
    sorted_var_inter_gpu_list = [x for _, x in sorted(zip(loss_list, var_inter_gpu_list))]
    sorted_loss_list = [x for x, _ in sorted(zip(loss_list, pipeline_split_list))]

    # for i in range(len(sorted_pipeline_split_list)):
    #     print(f"loss {sorted_loss_list[i]}")
    #     print(f" {sorted_pipeline_split_list[i]}")
    #     print(f"sorted_loss_list {sorted_loss_list[i]}")
    #     print(f"sorted_var_intra_gpu_list {sorted_var_intra_gpu_list[i]}")
    #     print(f"sorted_var_inter_gpu_list {sorted_var_inter_gpu_list[i]}")

    # exit()

    return sorted_pipeline_split_list






# 示例用法






def idx_to_gpu_type_num_dict(i,j, total_gpu_type_num_dict):

    gpu_type_num_dict = {}
    for k in range(i,j):
        gpu_type_num_dict[f"{k-i}"] = total_gpu_type_num_dict[f"{k}"]

    return gpu_type_num_dict



def split_devices_to_pipeline_large(input_dict, n):
    # 将输入字典按键的数值顺序排序，并提取对应的值列表
    sorted_keys = sorted(input_dict.keys(), key=lambda x: int(x))
    values = [input_dict[key] for key in sorted_keys]
    k = len(values)

    # 处理无法分割的情况
    if n <= 0 or n > k:
        return []

    # 生成所有可能的分割点组合
    split_combinations = itertools.combinations(range(k-1), n-1)
    all_splits = []
    all_splits_point  = []
    for split_points in split_combinations:
        current = 0
        split_result = []
        for point in split_points:
            split_result.append(values[current:point+1])
            current = point + 1
        # 添加最后一个子列表
        split_result.append(values[current:])
        all_splits.append(split_result)
        # 记录分割点
        all_splits_point.append(list(split_points))
    return all_splits






def precompute_all_k(n, max_k,cache_result,total_gpu_type_num_dict,hash_list_of_dicts):


    def fun(i,j):
        gpu_type_num_dict = idx_to_gpu_type_num_dict(i-1,j-1,total_gpu_type_num_dict)
        cache_prefix = hash_list_of_dicts(gpu_type_num_dict)
        if cache_prefix not in cache_result or cache_result[cache_prefix] == None:
            return -MAX_VALUE

        return cache_result[cache_prefix].throughput

    dp = {}  # dp[m][j] 表示将前j个元素分成m段的最大和
    split_history = {}  # split_history[m][j] 记录分割点

    # 初始化 m=1
    dp[1] = [0] * (n + 1)
    for j in range(1, n + 1):
        dp[1][j] = fun(1, j)
    split_history[1] = [0] * (n + 1)  # m=1时无需分割，但填充占位

    # 动态规划处理 m=2 到 max_k
    for m in range(2, max_k + 1):
        dp_current = [-float('inf')] * (n + 1)
        split_current = [0] * (n + 1)  # 分割点数组长度为n+1
        for j in range(m, n + 1):
            max_val = -float('inf')
            best_i = -1
            for i in range(1, j):
                current_sum = dp[m-1][i] + fun(i, j)
                if current_sum > max_val:
                    max_val = current_sum
                    best_i = i
            dp_current[j] = max_val
            split_current[j] = best_i
        dp[m] = dp_current
        split_history[m] = split_current  # 确保数组长度正确

    return dp, split_history

def get_results_for_ks(n, k_list, dp, split_history):
    results = {}
    for k in k_list:
        if k < 1 or k > len(dp):
            results[k] = (0, [])
            continue
        max_sum = dp[k][n]
        if k == 1:
            results[k] = (max_sum, [])
            continue

        # 回溯分割点：从 m=k 到 m=2
        split_points = []
        current_j = n
        for m in range(k, 1, -1):
            if m not in split_history:
                break
            split_array = split_history[m]
            if current_j < 1 or current_j > n:  # 检查索引合法性
                break
            split_i = split_array[current_j]
            if split_i < 1:  # 分割点至少为1
                break
            split_points.append(split_i)
            current_j = split_i

        split_points.reverse()  # 调整顺序为从左到右
        results[k] = (max_sum, split_points)
    return results


def split_to_cache_result(split, cache_result,total_gpu_type_num_dict,hash_list_of_dicts):
    prev = 0

    config_list = []
    for i in range(len(split)):
        gpu_type_num_dict = idx_to_gpu_type_num_dict(prev,split[i],total_gpu_type_num_dict)
        cache_prefix = hash_list_of_dicts(gpu_type_num_dict)
        if cache_prefix not in cache_result:
            return []
        prev = split[i]
        config_list.append(cache_result[cache_prefix])

    gpu_type_num_dict = idx_to_gpu_type_num_dict(prev, len(total_gpu_type_num_dict),total_gpu_type_num_dict)
    cache_prefix = hash_list_of_dicts(gpu_type_num_dict)
    if cache_prefix not in cache_result:
        return []
    config_list.append(cache_result[cache_prefix])

    return config_list


# def get_node_order_list_each_device_group(gpu_type_num_dict,device_group_list):

from collections import Counter
import copy

def group_into_blocks(lst):
    if not lst:
        return []
    blocks = []
    current = [lst[0]]
    for elem in lst[1:]:
        if elem == current[-1]:
            current.append(elem)
        else:
            blocks.append(current)
            current = [elem]
    blocks.append(current)
    return blocks

def merge_blocks(blocks):
    merged = []
    for block in blocks:
        if not merged:
            merged.append(list(block))
        else:
            last_block = merged[-1]
            if last_block[0] == block[0]:
                last_block.extend(block)
            else:
                merged.append(list(block))
    return merged

def is_valid(merged_blocks):
    seen = set()
    for block in merged_blocks:
        char = block[0]
        if char in seen:
            return False
        seen.add(char)
    return True

def generate_unique_permutations(type_list):
    blocks = group_into_blocks(type_list)
    all_perms = itertools.permutations(blocks)
    seen = set()
    result = []
    for perm in all_perms:
        merged = merge_blocks(perm)
        if is_valid(merged):
            combined = []
            for b in merged:
                combined.extend(b)
            combined_tuple = tuple(combined)
            if combined_tuple not in seen:
                seen.add(combined_tuple)
                result.append(list(combined_tuple))
    return result

def generate_valid_permutations(type_list_, num_list_ ,easy_mode = False):


    if(easy_mode):
        type_list = []
        total_res = []
        for key in type_list_.keys():
            type_list.append(json.dumps(type_list_[key]))
        res = generate_unique_permutations(type_list)
        res_ = []
        for item in res:
            item_ = {}
            idx = 0
            for i in item:
                # print(i)
                item_[f"{idx}"]=json.loads(i)
                idx+=1
            res_.append(item_)
        total_res.append(res_)
        return  total_res
    print(type_list_)
    type_list = []
    for key in type_list_.keys():
        type_list.append(json.dumps(type_list_[key]))
    # print(f"type_list {type_list}")
    # 统计各类型的总数量
    num_list__ = copy.deepcopy(num_list_)
    type_counts = Counter(type_list)
    # 检查总和是否符合
    total_res = []
    for num_list in num_list__ :
        num_list_temp = []
        sum_num_list = 0
        for i in range(len(num_list)):
            sum_num_list += num_list[i]
            if(sum_num_list >= type_list_['0']['GPU_NUM']):
                num_list_temp.append(sum_num_list//type_list_['0']['GPU_NUM'])
                sum_num_list = 0
        num_list = num_list_temp
        total = sum(num_list)
        if total != len(type_list):
            print(f"Error: The sum of num_list {total} does not match the total number of devices {len(type_list)}.")
            return []

        allocations = []
        num_regions = len(num_list)

        # 使用回溯法生成所有有效的类型分配方案
        def backtrack(region_idx, current_counts, current_allocation):
            if region_idx == num_regions:
                # 检查是否所有类型的数量都恰好用完
                if all(count == 0 for count in current_counts.values()):
                    allocations.append(current_allocation.copy())
                return

            required = num_list[region_idx]
            # 遍历所有可能的类型，其剩余数量足够当前区域的需求
            for type_name in list(current_counts.keys()):
                if current_counts[type_name] >= required:
                    new_counts = current_counts.copy()
                    new_counts[type_name] -= required
                    if new_counts[type_name] == 0:
                        del new_counts[type_name]
                    new_allocation = current_allocation.copy()
                    new_allocation.append(type_name)
                    backtrack(region_idx + 1, new_counts, new_allocation)

        backtrack(0, type_counts.copy(), [])

        # 生成所有可能的排列
        permutations = []
        seen = set()
        for alloc in allocations:
            perm = []
            for i in range(len(alloc)):
                perm += [alloc[i]] * num_list[i]
            # 转换为元组以去重
            perm_tuple = tuple(perm)
            if perm_tuple not in seen:
                seen.add(perm_tuple)
                permutations.append(list(perm_tuple))

        res_ = []

        for item in permutations:
            item_ = {}
            idx = 0
            for i in item:
                # print(i)
                item_[f"{idx}"]=json.loads(i)
                idx+=1
            res_.append(item_)
        total_res.append(res_)
    return total_res
