import os

def write_result_to_csv(result_dir, filename, metrics: list):
    with open(os.path.join(result_dir, f"{filename}.csv"),'a') as f:
        metrics = map(str, metrics) 
        f.write(",".join(metrics))
        f.write('\n')

def create_result_dir():
    i = 1
    while True:
        path = f"./runs/run_{i}"
        if not os.path.isdir(path):
            os.makedirs(path)
            break
        i += 1
    return path