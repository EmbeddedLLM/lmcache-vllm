from typing import List, Tuple, Any
import torch
#from vllm.distributed import broadcast_object_list, broadcast
#from vllm.sequence import SequenceGroupMetadata

'''
def broadcast_list(
        is_driver_worker: bool,
        lis: List[Any],
        device
    ) -> List[Any]:
    """
    Use vllm broadcast primitives to broadcast a list of object from driver worker to other workers
    """
    list_length_tensor = torch.tensor(len(lis), device=device)
    broadcast(list_length_tensor, src = 0)
    list_len = list_length_tensor.item()

    if list_len == 0:
        return []

    if not is_driver_worker:
        lis = [0] * list_len

    broadcast_object_list(lis, src = 0)
    return lis

def broadcast_input_ids_list(
        is_driver_worker: bool,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        device,
    ) -> List[torch.Tensor]:
    """
    Extract a list of input ids from sequence group metadata and broadcast it to other workers
    Each input ids is a integer tensor
    """
    input_ids_list = []
    if is_driver_worker:
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = seq_group_metadata.seq_data.keys()
            assert len(seq_ids) == 1
            seq_id = list(seq_ids)[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            whole_input_tokens = torch.tensor(seq_data.get_token_ids())
            input_ids_list.append(whole_input_tokens)
        broadcast_list(is_driver_worker, input_ids_list, device)
    else:
        input_ids_list = broadcast_list(is_driver_worker, input_ids_list, device)

    return input_ids_list

def broadcast_tokens_and_block_tables(
        is_driver_worker: bool,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        device,
    ) -> List[Tuple[List[int], List[int]]]:
    """
    Extract a list of input ids and block tables from sequence group metadata and broadcast it to other workers
    Both input ids and block table are list of integers
    """
    ret = []
    if is_driver_worker:
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            if seq_group_metadata.block_tables is not None:
                ret.append((seq_data.get_token_ids(), seq_group_metadata.block_tables[seq_id]))
        broadcast_list(is_driver_worker, ret, device)
        return ret
    else:
        return broadcast_list(is_driver_worker, ret, device)
'''
# TODO(Jiayi): error handling
def init_vllm_comm(
    backend: str, 
    world_size: int, 
    lmc_rank: int, 
    tp_ranks: List[List[int]],
    pp_ranks: List[List[int]],
    vllm_ranks: List[List[int]],
    world_ranks: List[List[int]],
    distributed_init_method: str):
    # Initialize vllm-related commuication here
    # Only related vllm, not lmcache
    torch.distributed.init_process_group(
        backend=backend,
        init_method=distributed_init_method,
        world_size=world_size,
        rank=lmc_rank)
    
    # TODO(Jiayi): TP/PP/World should be passed in as params
    # Initialize world group
    for ranks in vllm_ranks:
        device_group_world = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_world = torch.distributed.new_group(ranks, backend="gloo")
    
    # Initialize TP group
    for ranks in tp_ranks:
        device_group_TP = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_TP = torch.distributed.new_group(ranks, backend="gloo")
    
    # Initialize PP group
    for ranks in pp_ranks:
        device_group_PP = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_PP = torch.distributed.new_group(ranks, backend="gloo")
    
    '''
    for ranks in world_ranks:
        device_group = torch.distributed.new_group(ranks, backend=backend)
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
    '''