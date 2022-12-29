import os
from abc import ABC
from typing import Dict

import pandas as pd


class GroupGenerator(ABC):
    @classmethod
    def save_group_df(cls, data_frame: pd.DataFrame, output_dir: str, group_type: str, group_size: int, other_params: Dict = None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = f'{group_type}_{group_size}'
        if other_params:
            file_name += '_' + '_'.join([f'{key}={value}' for key, value in other_params.items()])

        file_path = os.path.join(output_dir, f'{file_name}.csv')
        print(f'Saving groups to {file_path}')
        data_frame.to_csv(file_path, index=False, header=False)
