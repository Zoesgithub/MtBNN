from loguru import logger
import os
import argparse
import importlib
import random
import numpy as np
from utils import DataGenerator, MutGenerator, GetSummaryStatisticsCallback, writeFile
import torch
from torch.utils.data import DataLoader


def main():

    # load config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="the path to the config file")
    args = parser.parse_args()
    config_path = args.config.replace("/", ".")
    config = importlib.import_module(config_path)
    config = config.config
    logger.add(config.log_path)

    # set seeds
    os.environ["PYTHONHASHSEED"] = str(1+config.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(1+config.seed)
    np.random.seed(1+config.seed)
    torch.random.manual_seed(1+config.seed)
    print("finish set env")

    if config.state=="pretrain":
        # prepare model
        train_data = DataLoader(DataGenerator(taskList=config.tasklist, path=config.path, endfix="_train"),
                                shuffle=True, batch_size=config.batch_size, num_workers=0, drop_last=True)
        validation_data = DataLoader(DataGenerator(taskList=config.tasklist, path=config.path, endfix="_valid"), shuffle=False,
                                     batch_size=config.batch_size, num_workers=0, drop_last=True)
        test_data = DataLoader(DataGenerator(taskList=config.tasklist, path=config.path, endfix="_test"), shuffle=False,
                                     batch_size=config.batch_size, num_workers=0, drop_last=True)



        summary = GetSummaryStatisticsCallback(
            config.model,
            train_data, validation_data, test_data=test_data,
            model_save_path=os.path.join(config.save_path, "models"),
            ispretrain=True
        )

        logger.info("Finish loading data ...")
        logger.info(
            "Train data size {} validation data size {} test data size {}".format(len(train_data), len(validation_data),
                                                                                  len(test_data)))

        summary.fit(config.epoch_num)
        logger.info("Finish training, start eval ...")
        config.model.load_state_dict(torch.load(os.path.join(config.save_path, "models", "model.ckpt-best")))
        writeFile(config.model.eval(validation_data), os.path.join(config.save_path, "valid_"))
        writeFile(config.model.eval(test_data), os.path.join(config.save_path, "test_"))
        return

    elif config.state=="cv": ## five-fold cross validation
        # load model
        mut_data = MutGenerator(config.trainjsonfile, config.taskname, config.tasklist)
        summary = GetSummaryStatisticsCallback(
            config.model,
            None, None, test_data=None,
            model_save_path=os.path.join(config.save_path, "models"),
            ispretrain=False
        )
        summary.cross_validation(config.epoch_num, mut_data, config.batch_size, config.load_path, num_fold=5, prefix=config.trainjsonfile.split("/")[-1])

    elif config.state=="ft": ## fine tuning
        mut_data = MutGenerator(config.trainjsonfile, config.taskname, config.tasklist)
        summary = GetSummaryStatisticsCallback(
            config.model,
            None, None, test_data=None,
            model_save_path=os.path.join(config.save_path, config.trainjsonfile.split("/")[-1]),
            ispretrain=False
        )
        summary.fine_tuning(config.epoch_num, mut_data, config.batch_size, config.load_path)
    elif config.state=="evmut": ## eval mutations
        if not isinstance(config.trainjsonfile, list):
            config.trainjsonfile=[config.trainjsonfile]
        for data in config.trainjsonfile:
            mut_data = DataLoader(MutGenerator(data, config.taskname, config.tasklist), batch_size=config.batch_size)
            config.model.load_state_dict(torch.load(config.load_path))
            writeFile(config.model.eval(mut_data, ismut=True), os.path.join(config.savepath, config.trainjsonfile.split("/")[-1]+"eval"))
    elif config.state=="eval": ## eval seq
        test_data=DataLoader(DataGenerator(taskList=config.tasklist, path=config.path, endfix="_test"), shuffle=False,
                                     batch_size=config.batch_size, num_workers=0, drop_last=False)
        config.model.load_state_dict(torch.load(config.load_path))
        writeFile(config.model.eval(test_data), os.path.join(config.save_path, "test_"))
    else:
        assert False, "state must in [evmut, cv, ft, pretrain]"





if __name__ == "__main__":
    main()
