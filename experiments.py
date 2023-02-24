from typing import Iterable
from tqdm import tqdm
from SCG import SCG_NQP, SCG_Yahoo

class Experiments:
    def __init__(self, method:str, dataset:str) -> None:
        self.method = method
        self.dataset = dataset

        self.setup()

    def setup(self):
        if self.method == 'scg':
            if self.dataset == 'NQP':
                self.experiment = SCG_NQP

    def run(self, iteration:int, repeat:int, args, to_file=True, path='Results/'):
        results = []

        for i in tqdm(range(repeat)):
            ep = self.experiment(args)
            values = ep.train()
            results.append(values)

        
