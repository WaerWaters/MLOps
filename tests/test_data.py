from data.get_data import Data


def test_data_load():
    data = Data(path="data/tiny-imagenet")
    assert len(data.get_train_val_test_sets(splits=[0.7, 0.1, 0.2])) == 3
