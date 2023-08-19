BATCH_SIZE = 128
EPOCHS = 100
momentum = 0.9

seed_size = 0.01
al_steps = 10

dataset = 'compas' # celebA, compas
method = 'mo-al' # ed, rs-b, rs, us, mo-ed, al, fp-o, mo-al

if method == 'fp-o':
    lambd_al = 1

if dataset == 'celebA':
    learning_rate = 0.001
    train_datapath = 'data/celebA/train.csv'
    val_datapath = 'data/celebA/val.csv'
    test_datapath = 'data/celebA/test.csv'
    del_size = 3200
    lambd_in = 0.85
    lambd_al = 0.15
elif dataset == 'compas':
    learning_rate = 0.03
    train_datapath = 'data/compas/compas_processed.csv'
    val_datapath = None
    test_datapath = None
    del_size = 190
    lambd_in = 0.85
    lambd_al = 0.15
