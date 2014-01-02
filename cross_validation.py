
def train_test_split(data, n_train):
    if isinstance(n_train, float):
        m = data.shape[0]
        n_train = int(m * n_train)
    
    if not isinstance(n_train, int):
        raise ValueError('n_train is not of type int or float')
        
    return (data[:n_train], data[n_train:])
