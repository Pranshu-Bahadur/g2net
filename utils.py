import pandas as pd
from multiprocessing import Pool
from os import walk, listdir
from functools import reduce
import os
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor, stack
from numpy import load, hstack
import numpy as np

def multi_pathfinder(root_dir : str, num_cpus : int) -> pd.DataFrame:
    start = time.time()
    dirs = [f"{root_dir}/{d}" for d in listdir(root_dir) if os.path.isdir(f"{root_dir}/{d}")]
    with Pool(num_cpus) as threads:
        dfs = threads.map(_walker, dirs)
    df = pd.concat(dfs)
    stop = time.time()
    print(f"\n\n========\nTotal Time taken :\n {stop - start}\n========")
    print(df.head())
    return df


def _walker(directory : str) -> pd.DataFrame:
    functions = [(filter,lambda x: len(x[2]) > 0), (map, lambda x: (x[0], x[2]))]
    paths_and_files = list(reduce(lambda x,f: f[0](f[1], x), functions, walk(directory)))
    list_dfs = list(map(lambda x: pd.DataFrame.from_dict(list(map(lambda y: {"id": y.split(".")[0], "path": f"{x[0]}/{y}"}, x[1])), orient='columns'), tqdm(paths_and_files)))
    df = pd.concat(list_dfs) if len(list_dfs) > 0 else pd.DataFrame()
    return df


class NumpyImagesCSVDataset(Dataset):
    def __init__(self, root_dir : str, path_to_csv : str, is_train : bool, transforms):
        self.df = pd.read_csv(path_to_csv).join(multi_pathfinder(root_dir, os.cpu_count()).set_index('id'), on="id") if is_train else multi_pathfinder(root_dir, os.cpu_count())
        self.is_train = is_train
        #self.distribution = self.df.groupby(['target'], axis=0).apply(lambda x: x.count())['target'].values
        #print(self.distribution)
        self.transforms = transforms

    def __getitem__(self, index : int):
        data = self.df.iloc[index].to_dict()
        waves = from_numpy(hstack(load(data['path'])))#hstack(l
        waves /= waves.max()
        x = waves.float()
        y = tensor(data['target']) if self.is_train else tensor(0)
        if self.is_train:
            return x, y
        return x, data["id"]

    def __len__(self):
        return len(self.df)
















"""
given - list_abs_path : List[str]

return most_suitable : str
"""
def df_from_abs_paths(abs_paths) -> str :
    df = pd.DataFrame(abs_paths)
    df = df['0'].str.split('/', expand=True)
    return df.groupby(list(df.columns), axis=1).apply(lambda x: x.count()).max().index.to_list()[0]


