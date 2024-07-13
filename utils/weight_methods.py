import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union, Any
import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch import Tensor

from utils.min_norm_solvers import MinNormSolver, gradient_normalizers


# compared with original function, line 223,226 is modified


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

    @abstractmethod
    def get_weighted_loss(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ],
            last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
            representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
            **kwargs,
    ):
        pass

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            last_shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            relative_loss_drop: Union[List[float], None] = None,
            optimizer=None,
            **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, _ = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        loss.backward()
        return loss, None

    def __call__(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class NashMTL(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            max_norm: float = 1.0,
            update_weights_every: int = 1,
            optim_niter=20,
    ):
        super(NashMTL, self).__init__(n_tasks=n_tasks, device=device, )

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg, alpha_t):
        return (
                (self.alpha_param.value is None)
                or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
                or (
                        np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                        < 1e-6
                )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.n_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.n_tasks, self.n_tasks), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(self, losses, shared_parameters, **kwargs, ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :

        Returns
        -------
        """

        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            grads = {}
            for i, loss in enumerate(losses):
                g = list(
                    torch.autograd.grad(
                        loss,
                        shared_parameters,
                        retain_graph=True,
                    )
                )
                grad = torch.cat([torch.flatten(grad) for grad in g])
                grads[i] = grad

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (torch.norm(GTG).detach().cpu().numpy().reshape((1,)))
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            # don't update
            self.step += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        return weighted_loss, extra_outputs

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            last_shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        loss, extra_outputs = self.get_weighted_loss(losses=losses, shared_parameters=shared_parameters, **kwargs, )
        loss.backward()

        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        return loss, extra_outputs


class AlignedMTL(WeightMethod):
    def __init__(self, n_tasks, device, scale_mode='min', scale_decoder_grad=False, **kwargs):
        super().__init__(n_tasks, device=device)
        self.scale_decoder_grad = scale_decoder_grad
        self.scale_mode = scale_mode
        print('AMGDA balancer scale mode:', self.scale_mode)

    @staticmethod
    def ProcrustesSolver(grads, scale_mode='min'):
        assert (
                len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        with torch.no_grad():
            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

            singulars, basis = torch.linalg.eigh(cov_grad_matrix_e)
            tol = (
                    torch.max(singulars)
                    * max(cov_grad_matrix_e.shape[-2:])
                    * torch.finfo().eps
            )
            rank = sum(singulars > tol)

            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            if scale_mode == 'min':
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            elif scale_mode == 'median':
                weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
            elif scale_mode == 'rmse':
                weights = basis * torch.sqrt(singulars.mean())

            weights = weights / torch.sqrt(singulars).view(1, -1)
            weights = torch.matmul(weights, basis.T)
            grads = torch.matmul(grads, weights.unsqueeze(0))

            return grads, weights, singulars

    def get_weighted_loss(self, losses, shared_parameters, task_specific_parameters, shared_representation=None,
                          last_shared_layer_params=None, **kwargs):
        grads = {}
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            grads[i] = grad

        G = torch.stack(tuple(v for v in grads.values()))

        grads, weights, singulars = self.ProcrustesSolver(G.T.unsqueeze(0), self.scale_mode)
        grad, weights = grads[0].sum(-1), weights.sum(-1)

        weighted_loss = sum([losses[i] * weights[i] for i in range(len(weights))])
        return weighted_loss, None


class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)


class ScaleInvariantLinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(torch.log(losses) * self.task_weights)
        return loss, dict(weights=self.task_weights)


class MGDA(WeightMethod):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """

    def __init__(
            self, n_tasks, device: torch.device, params="shared", normalization="none"
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def get_weighted_loss(
            self,
            losses,
            shared_parameters=None,
            last_shared_parameters=None,
            representation=None,
            **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        last_shared_parameters :
        representation :
        kwargs :

        Returns
        -------

        """
        # Our code
        grads = {}
        params = dict(
            rep=representation, shared=shared_parameters, last=last_shared_parameters
        )[self.params]
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                )
            )
            # Normalize all gradients, this is optional and not included in the paper.

            grads[i] = [torch.flatten(grad) for grad in g]

        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))]
        )
        sol = sol * self.n_tasks  # make sure it sums to self.n_tasks
        weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])

        return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))


class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]

        return loss, dict(weights=self.weights)


class Uncertainty(WeightMethod):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(
            [
                0.5 * (torch.exp(-logs) * loss + logs)
                for loss, logs in zip(losses, self.logsigma)
            ]
        )

        return loss, dict(
            weights=torch.exp(-self.logsigma)
        )  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]


class PCGrad(WeightMethod):
    """Modification of: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

    @misc{Pytorch-PCGrad,
      author = {Wei-Cheng Tseng},
      title = {WeiChengTseng/Pytorch-PCGrad},
      url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
      year = {2020}
    }

    """

    def __init__(self, n_tasks: int, device: torch.device, reduction="sum"):
        super().__init__(n_tasks, device=device)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def get_weighted_loss(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        raise NotImplementedError

    def _set_pc_grads(self, losses, shared_parameters, task_specific_parameters=None):
        # shared part
        shared_grads = []
        for l in losses:
            shared_grads.append(
                torch.autograd.grad(l, shared_parameters, retain_graph=True)
            )

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        non_conflict_shared_grads = self._project_conflicting(shared_grads)
        for p, g in zip(shared_parameters, non_conflict_shared_grads):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(), task_specific_parameters
            )
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = sum(
                    [torch.dot(torch.flatten(grad_i), torch.flatten(grad_j)) for grad_i, grad_j in zip(g_i, g_j)])
                if g_i_g_j < 0:
                    g_j_norm_square = (torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2)
                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= g_i_g_j * grad_j / g_j_norm_square

        merged_grad = [sum(g) for g in zip(*pc_grad)]
        if self.reduction == "mean":
            merged_grad = [g / self.n_tasks for g in merged_grad]

        return merged_grad

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        self._set_pc_grads(losses, shared_parameters, task_specific_parameters)
        return None, {}  # NOTE: to align with all other weight methods


class OurMethod_III(WeightMethod):
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.step = 0
        self.grad_buffer = None

    def get_weighted_loss(self, losses: torch.Tensor,
                          shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
                          task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
                          relative_loss_drop: Union[List[float], None] = None,
                          **kwargs,):
        return torch.log(losses), None

    @staticmethod
    def _flatten_grad(shared_grads):
        grad, shape = [], []
        for p in shared_grads:
            shape.append(p.shape)
            grad.append(torch.flatten(p))
        return torch.cat(grad), shape

    @staticmethod
    def _unflatten_grad(grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _set_grads(self, losses, shared_parameters, task_specific_parameters=None, optimizer=None,
                   relative_loss_drop=None, beta=0.5):

        shared_grads, flatten_grads, shapes = [], [], []
        for i, l in enumerate(losses):
            g = torch.autograd.grad(l, shared_parameters, retain_graph=True)
            shared_grads.append(g)
            flatten_grad, shape = self._flatten_grad(shared_grads[i])
            flatten_grads.append(flatten_grad)
            shapes.append(shape)

        flatten_grads = torch.stack(flatten_grads, dim=0)
        # 过拟合了，drop会变小，正常的时候drop会变大, 让正常的那一组的beta变得很小
        relative_loss_drop = torch.sigmoid(torch.tensor(relative_loss_drop,
                                                        dtype=torch.float32, device=flatten_grads.device))
        # what if the both tasks are over fit, the drop all decrease?
        beta = (beta ** relative_loss_drop).view(self.n_tasks, 1)
        if self.step == 0:
            self.grad_buffer = flatten_grads
        else:
            # when step increase, more trust on new grads, but this is not reasonable when overfit happen?
            self.grad_buffer = beta*self.grad_buffer+(1-beta)*flatten_grads

        self.step += 1
        # flatten_grads = self._project_conflicting()
        flatten_grads = self.grad_buffer
        grads_norm = flatten_grads.norm(dim=-1)
        alpha = grads_norm.max()/(grads_norm + 1e-8)
        # flatten_grads = [relative_loss_drop[i]*alpha[i]*self.grad_buffer[i] for i in range(self.n_tasks)]
        flatten_grads = [alpha[i] * flatten_grads[i] for i in range(self.n_tasks)]

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        flatten_grads = torch.stack(flatten_grads, dim=0)  # shape [2, 266880]
        flatten_grads = flatten_grads.sum(0)
        unflatten_grad = self._unflatten_grad(flatten_grads, shapes[0])
        for p, g in zip(shared_parameters, unflatten_grad):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(), task_specific_parameters
            )
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

        return beta

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            optimizer=None,
            relative_loss_drop: Union[List[float], None] = None,
            **kwargs,
    ):
        losses, _ = self.get_weighted_loss(losses, relative_loss_drop=relative_loss_drop)
        beta = self._set_grads(losses, shared_parameters, task_specific_parameters, optimizer, relative_loss_drop)
        # default: l2 = -1,
        l2_param = torch.sigmoid(torch.tensor(-sum(relative_loss_drop)/len(relative_loss_drop),
                                dtype=torch.float32, device=losses.device)) * 1e-3
        optimizer.param_groups[0]['weight_decay'] = l2_param
        # 过拟合时，relative drop会变小， 提高l2 regularization
        return None, beta  # NOTE: to align with all other weight methods


class DB_MTL(WeightMethod):
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.step = 0
        self.grad_buffer = None

    def get_weighted_loss(self, losses: torch.Tensor,
                          shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
                          task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
                          **kwargs,):
        return torch.log(losses), None

    @staticmethod
    def _flatten_grad(shared_grads):
        grad, shape = [], []
        for p in shared_grads:
            shape.append(p.shape)
            grad.append(torch.flatten(p))
        return torch.cat(grad), shape

    @staticmethod
    def _unflatten_grad(grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _set_grads(self, losses, shared_parameters, task_specific_parameters=None, optimizer=None,
                   relative_loss_drop=None, beta=0.5):
        shared_grads, flatten_grads, shapes = [], [], []
        for i, l in enumerate(losses):
            g = torch.autograd.grad(l, shared_parameters, retain_graph=True)
            shared_grads.append(g)
            flatten_grad, shape = self._flatten_grad(shared_grads[i])
            flatten_grads.append(flatten_grad)
            shapes.append(shape)

        flatten_grads = torch.stack(flatten_grads, dim=0)
        if self.step == 0:
            self.grad_buffer = flatten_grads
        else:
            # when step increase, more trust on new grads, but this is not reasonable when overfit happen?
            self.grad_buffer = beta*self.grad_buffer+(1-beta)*flatten_grads

        self.step += 1
        grads_norm = self.grad_buffer.norm(dim=-1)
        alpha = grads_norm.max()/(grads_norm + 1e-8)
        flatten_grads = [alpha[i]*self.grad_buffer[i] for i in range(self.n_tasks)]
        # grad_weights = F.softmax(torch.tensor(relative_loss_drop), dim=0)
        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        flatten_grads = torch.stack(flatten_grads, dim=0)  # shape [2, 266880]
        flatten_grads = flatten_grads.sum(0)
        unflatten_grad = self._unflatten_grad(flatten_grads, shapes[0])
        for p, g in zip(shared_parameters, unflatten_grad):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(losses.sum(), task_specific_parameters)
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            optimizer=None,
            relative_loss_drop: Union[List[float], None] = None,
            **kwargs,
    ):
        losses, _ = self.get_weighted_loss(losses)
        self._set_grads(losses, shared_parameters, task_specific_parameters, optimizer, relative_loss_drop)
        return None, {}  # NOTE: to align with all other weight methods


class GradDrop(WeightMethod):
    """
    GradDrop
    """

    def __init__(self, n_tasks: int, device: torch.device, reduction="sum"):
        super().__init__(n_tasks, device=device)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def get_weighted_loss(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        raise NotImplementedError

    @staticmethod
    def _flatten_grad(shared_grads):
        grad, shape = [], []
        for p in shared_grads:
            shape.append(p.shape)
            grad.append(torch.flatten(p))
        return torch.cat(grad), shape

    @staticmethod
    def _unflatten_grad(grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _set_drop_grads(self,
                        losses,
                        shared_parameters, task_specific_parameters=None,
                        leak=0):
        # shared part
        shared_grads, flatten_grads, shapes = [], [], []
        for i, l in enumerate(losses):
            shared_grads.append(
                torch.autograd.grad(l, shared_parameters, retain_graph=True)
            )
            flatten_grad, shape = self._flatten_grad(shared_grads[i])
            flatten_grads.append(flatten_grad)
            shapes.append(shape)

            # shared_grads.append(torch.cat([grad.view(-1) for grad in g if grad is not None]))
            # flatten the gradients, 266880
            # shared_grads.append(torch.cat([torch.flatten(grad) for grad in g if grad is not None]))

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        flatten_grads = torch.stack(flatten_grads, dim=0)  # shape [2, 266880]
        P = 0.5 * (1 + flatten_grads.sum(0) / (flatten_grads.abs().sum(0) + 1e-8))  # p have the same shape
        U = torch.randn_like(P)
        M = P.gt(U).unsqueeze(0).repeat(self.n_tasks, 1) * flatten_grads.gt(0) + \
            P.lt(U).unsqueeze(0).repeat(self.n_tasks, 1) * flatten_grads.lt(0)
        transformed_grad = (flatten_grads * (leak + (1 - leak) * M)).sum(0)
        transformed_grad = self._unflatten_grad(transformed_grad, shapes[0])
        for p, g in zip(shared_parameters, transformed_grad):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(), task_specific_parameters
            )
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            representation=None,
            optimizer=None,
            **kwargs,
    ):
        self._set_drop_grads(losses, shared_parameters, task_specific_parameters)
        return None, {}  # NOTE: to align with all other weight methods


class FAMO(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            gamma: float = 1e-5,
            w_lr: float = 0.025,
            task_weights: Union[List[float], torch.Tensor] = None,
            max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm

    def set_min_losses(self, losses):
        self.min_losses = losses  # min_losses computed on the train data after gradient update.

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        # update never run in our experiment
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()


class CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4):
        super().__init__(n_tasks, device=device)
        self.c = c

    def get_weighted_loss(
            self,
            losses,
            shared_parameters,
            **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < (self.n_tasks - 1):
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                    x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                    + c
                    * np.sqrt(
                x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                + 1e-8
            )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha ** 2)
        else:
            return g / (1 + alpha)

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        self.get_weighted_loss(losses, shared_parameters)
        return None, {}  # NOTE: to align with all other weight methods


class RLW(WeightMethod):
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)


class UniWeight(WeightMethod):
    """
    uniform baseline
    """

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    @staticmethod
    def _flatten_grad(shared_grads):
        grad, shape = [], []
        for p in shared_grads:
            shape.append(p.shape)
            grad.append(torch.flatten(p))
        return torch.cat(grad), shape

    def get_weighted_loss(self, losses: torch.Tensor, shared_parameters, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (torch.ones(self.n_tasks)).to(self.device)
        loss = torch.sum(losses * weight)
        flatten_grads, grads_norm, cosine_sim = [], [], []
        for i, l in enumerate(losses):
            g = torch.autograd.grad(l, shared_parameters, retain_graph=True)
            flatten_grad, shape = self._flatten_grad(g)
            flatten_grads.append(flatten_grad)
            grads_norm.append(flatten_grad.norm().cpu())

        cosine_sim = torch.dot(flatten_grads[0], flatten_grads[1])/(grads_norm[0]*grads_norm[1])
        grads_norm.append(cosine_sim.cpu())
        return loss, grads_norm

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            last_shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            relative_loss_drop: Union[List[float], None] = None,
            optimizer=None,
            **kwargs,
    ) -> Tuple[Tensor, List[Any]]:
        loss, grads_info = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        loss.backward()
        return loss, grads_info


class IMTLG(WeightMethod):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(
            self,
            losses,
            shared_parameters,
            **kwargs,
    ):
        grads = {}
        norm_grads = {}

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        G = torch.stack(tuple(v for v in grads.values()))
        D = (
                G[
                    0,
                ]
                - G[
                  1:,
                  ]
        )

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = (
                U[
                    0,
                ]
                - U[
                  1:,
                  ]
        )
        first_element = torch.matmul(
            G[
                0,
            ],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.n_tasks - 1, device=self.device) * 1e-8
                + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat(
            (torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_)
        )

        loss = torch.sum(losses * alpha)

        return loss, dict(weights=alpha)


class DynamicWeightAverage(WeightMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
            self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[
                                                                 : self.iteration_window, :
                                                                 ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
            self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(
    stl=STL,
    ls=LinearScalarization,
    uw=Uncertainty,
    pcgrad=PCGrad,
    mgda=MGDA,
    cagrad=CAGrad,
    nashmtl=NashMTL,
    scaleinvls=ScaleInvariantLinearScalarization,
    rlw=RLW,
    imtl=IMTLG,
    dwa=DynamicWeightAverage,
    uniw=UniWeight,
    graddrop=GradDrop,
    famo=FAMO,
    alignmtl=AlignedMTL,
    dbmtl=DB_MTL,
    our_method=OurMethod_III
)
