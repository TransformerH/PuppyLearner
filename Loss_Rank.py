# pt_pre : s scale prediction probability on the correct category labels t
# pt_next: s+1 scale prediction probability on the correct category labels t
def loss_rank(pt_pre, pt_next):
    margin = 0
    return max(0, pt_pre - pt_next + margin)