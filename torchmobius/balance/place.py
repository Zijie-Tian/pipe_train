from typing import Dict, List
import itertools

def placement(shared_pcie: Dict, shared_pcie_n: int):
    
    def distance(shared_pcie: Dict, shared_pcie_n: int, devices: List):
        dis = 0
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                if shared_pcie[devices[i]] == shared_pcie[devices[j]]:
                    dis += (j - i)
        return dis
    
    max_dis = 0
    rc_device = []            
    for i in itertools.permutations(shared_pcie.keys()):
        dis = distance(shared_pcie, shared_pcie_n, list(i)) 
        if dis > max_dis:
            rc_device = list(i)
            max_dis = dis 
            
    
    return rc_device 