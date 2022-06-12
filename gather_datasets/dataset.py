from abc import ABC, abstractmethod
 
class Dataset(ABC):
    @abstractmethod
    def download_dataset(self, data_dir):
        pass

    @abstractmethod
    def process_dataset(self, data_dir, destination_dir, compress=True):
        pass
