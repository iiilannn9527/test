import torch
import torch.nn.functional as F
import logging
from collections import OrderedDict
# import matplotlib.pyplot as plt

class  DoubleSVD:
    def __init__(self, proj_rank, rank, device, decay):
        self.proj_rank = proj_rank
        self.rank = rank
        self.device = device
        self.sketch_matrix = None
        self.projector_matrix = None
        self.decay_factor = decay
        
    def increment_update(self, matrix):
        U, S, _ = torch.linalg.svd(matrix, full_matrices=False)
        self.projector_matrix = U[:, :self.proj_rank]
        
        if self.sketch_matrix is None:
            self.sketch_matrix = U[:, :self.rank] @ torch.diag(S[:self.rank])
        else:
            M = U[:, :self.rank] @ torch.diag(S[:self.rank]) # new Q
            SM = torch.cat([self.sketch_matrix, M], dim=1)
            U1, S1, _ = torch.linalg.svd(SM, full_matrices=False)
            Sr1 = S1[:self.rank+1]
            SS1 = torch.sqrt(torch.maximum(Sr1**2 - Sr1[self.rank]**2, torch.tensor(0.0)))
            self.sketch_matrix = U1[:, :self.rank] @ torch.diag(SS1[:self.rank])
            self.sketch_sigma = SS1[:self.rank]
            self.sketch_project_matrix = U1[:, :self.rank]
    
    def get_sketch_project_matrix(self):
        return self.sketch_project_matrix
    
    def get_projector_matrix(self):
        return self.projector_matrix
    
    def get_sketch_sigma(self):
        return self.sketch_sigma      

class CoSOProjector:
    def __init__(self, proj_rank, rank, scale=1.0, db_decay=0.5, update_proj_gap=1, n_tasks=10, cur_task=0):
        self.proj_rank = proj_rank
        self.rank = rank
        self.scale = scale
        self.db_decay = db_decay
        self.update_proj_gap = update_proj_gap
        # orthogonal projection
        self.DBSVD = None
        self.fd_project_matrix = None
        self.full_former_task_proj = None
        self.historical_projection = None
        
        self.n_tasks = n_tasks
        self.cur_task = cur_task
        
    def project(self, full_rank_grad, iter):
        
        # orthogonal project
        if self.cur_task > 0:
            full_rank_grad = full_rank_grad - self.full_former_task_proj @ (self.full_former_task_proj.T @ full_rank_grad)
            
        # frequent directions
        if self.fd_project_matrix is None or iter % self.update_proj_gap == 0:
            if self.DBSVD is None:
                self.DBSVD = DoubleSVD(proj_rank=self.proj_rank, rank=self.rank, device=full_rank_grad.device, decay=self.db_decay)
            self.DBSVD.increment_update(full_rank_grad)
            self.fd_project_matrix = self.DBSVD.get_projector_matrix()

        low_rank_grad = torch.matmul(self.fd_project_matrix.t(), full_rank_grad)
        
        return low_rank_grad

    def project_back(self, low_rank_grad):
        
        full_rank_grad = torch.matmul(self.fd_project_matrix, low_rank_grad)
        
        return full_rank_grad

    def update_projector(self, cur_task):
        self.cur_task = cur_task
        self.DBSVD = None
        self.fd_project_matrix = None
    
    def update_historical_space(self, threshold):
        
        sketch_sigma = self.DBSVD.get_sketch_sigma()
        sval_total = (sketch_sigma**2).sum()
        sval_ratio = (sketch_sigma**2) / sval_total
        rank_num = torch.sum(torch.cumsum(sval_ratio, dim=0) < threshold) # thereshold
        
        if self.full_former_task_proj is None:            
            self.full_former_task_proj = self.DBSVD.get_sketch_project_matrix()[:, :rank_num]
        else:
            self.full_former_task_proj = torch.cat([self.full_former_task_proj, self.DBSVD.get_sketch_project_matrix()[:, :rank_num]], dim=1)

        logging.info(f"Current dimension of project matrix: {self.full_former_task_proj.shape}")