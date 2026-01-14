import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CoSOVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

from utils.scheduler import CosineSchedule
from coso.adamw import CoSOAdamW
import wandb

num_workers = 16

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CoSOVitNet(args, True)
        logging.info("Use CoSO to train the network ...")

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        trainable_suffixes = ["attn.proj.weight", f"classifier_pool.{self._cur_task}"]
        
        # attn.proj param should use CoSO, and classifier_fc parameter should use amdaw
        coso_target_params = ["attn.proj"]
        
        coso_params = []
        for module_name, module in self._network.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(target_key in module_name for target_key in coso_target_params):
                continue
            # logging.info(f"coso module name ==> {module_name}")
            coso_params.append(module.weight)
        
        id_coso_params = [id(p) for p in coso_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in self._network.parameters() if id(p) not in id_coso_params]
        param_groups = [{'params': regular_params, 'lr': self.args["fc_lrate"]},
                        {'params': coso_params, 'lr': self.args["lrate"], 'proj_rank': self.args["proj_rank"], 'rank': self.args["rank"], 'scale': 1.0, 
                         'db_decay': 1.0, 'update_proj_gap': self.args["update_gap"]}]

        # freeze all parameters except coso parameters
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            # check the parameter contains trainable_suffixes
            # only set require grad for current task parameters
            if any(suffix in name for suffix in trainable_suffixes):
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        logging.info(f"Parameters to be updated: {sorted(list(enabled))}")
        
        if self._cur_task == 0:
            self.optimizer = CoSOAdamW(param_groups, weight_decay=self.args["weight_decay"], betas=(0.9,0.999))
            scheduler = CosineSchedule(self.optimizer, self.args["epochs"])
            self.train_function(train_loader, test_loader, self.optimizer, scheduler)
        else:
            self.optimizer.resetting_lr(self.args["lrate"], self.args["fc_lrate"])
            # reset optimizer state
            for param_group in self.optimizer.param_groups:
                if "rank" in param_group:
                    logging.info("resetting optimizer state for task {}".format(self._cur_task))
                    for p in param_group["params"]:
                        state = self.optimizer.state[p]
                        projector = state["projector"]
                        # update project matrix and projector
                        projector.update_historical_space(self.args["threshold"])
                        projector.update_projector(self._cur_task)
                        # update state
                        state['step'] = 0
                        state.pop('exp_avg')
                        state.pop('exp_avg_sq')
            logging.info("resetting done for task {}".format(self._cur_task))
            scheduler = CosineSchedule(self.optimizer, self.args["epochs"])
            self.train_function(train_loader, test_loader, self.optimizer, scheduler)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            one_epoch_steps = 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes
                
                logits = self._network(inputs)["logits"] / self.args["temperature"]
                loss = F.cross_entropy(logits, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                one_epoch_steps += 1

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # test_acc = self._compute_accuracy(self._network, test_loader)
            test_acc = 0
            
            info = "Task {}, Epoch {}/{}, Steps {} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["epochs"],
                one_epoch_steps,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            
            wandb.log({f"Task {self._cur_task} train epoch loss": losses / len(train_loader)})
            wandb.log({f"Task {self._cur_task} train acc": train_acc})
            wandb.log({f"Task {self._cur_task} test acc": test_acc})

            prog_bar.set_description(info)

        logging.info(info)
