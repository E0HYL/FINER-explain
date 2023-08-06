
# output time and msg for each function call in the test file
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper


def save_background(bg_path, train_x, train_y, num_sample=100):
    import numpy as np
    rint = np.random.choice(len(train_x), (num_sample), replace=False)
    bg_x = train_x[rint]

    train_x0 = train_x[train_y.argmax(1) == 0] # benign
    rint0 = np.random.choice(len(train_x0), (num_sample), replace=False)
    bg_x0 = train_x0[rint0]
    
    np.save(bg_path, bg_x) # for explainer
    np.save(bg_path.replace('.npy', '_0.npy'), bg_x0) # for feature reduction