from .lnaive_scheduler import lnaive_budget_scheduler
from .greedy_scheduler import greedy_budget_scheduler




def run_scheduler(
    node_path_list,
    importance_list,
    scheduler_name,
    bounces,
    C_total,
    network_generator,
    return_details=True,  
):
    if scheduler_name == "LNaive":
        return lnaive_budget_scheduler(
            node_path_list, importance_list, bounces, C_total, network_generator,
            return_details=return_details,   
        )
    elif scheduler_name == "Greedy":
        return greedy_budget_scheduler(
            node_path_list, importance_list, bounces, C_total, network_generator,
            return_details=return_details,   
        )
    raise ValueError(f"Unknown scheduler name: {scheduler_name}")