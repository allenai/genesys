import copy
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import sympy
import torch
import torch.nn as nn
from tqdm import tqdm

from .KANLayer import KANLayer
from .LBFGS import LBFGS
from .spline import curve2coef
from .Symbolic_KANLayer import SYMBOLIC_LIB, Symbolic_KANLayer


class KAN(nn.Module):
    """
    KAN class

    Attributes:
    -----------
        biases: a list of nn.Linear()
            biases are added on nodes (in principle, biases can be absorbed
            into activation functions. However, we still have them for better optimization)
        act_fun: a list of KANLayer
            KANLayers
        depth: int
            depth of KAN
        width: list
            number of neurons in each layer. e.g., [2,5,5,3] means 2D inputs,
            5D outputs, with 2 layers of 5 hidden neurons.
        grid: int
            the number of grid intervals
        k: int
            the order of piecewise polynomial
        base_fun: fun
            residual function b(x). an activation function
            phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun: a list of Symbolic_KANLayer
            Symbolic_KANLayers
        symbolic_enabled: bool
            If False, the symbolic front is not computed (to save time).
            Default: True.

    Methods:
    --------
        __init__():
            initialize a KAN
        initialize_from_another_model():
            initialize a KAN from another KAN (with the same shape, but
            potentially different grids)
        update_grid_from_samples():
            update spline grids based on samples
        initialize_grid_from_another_model():
            initalize KAN grids from another KAN
        forward():
            forward
        set_mode():
            set the mode of an activation function: 'n' for numeric, 's' for
            symbolic, 'ns' for combined (note they are visualized differently
            in plot(). 'n' as black, 's' as red, 'ns' as purple).
        fix_symbolic():
            fix an activation function to be symbolic
        suggest_symbolic():
            suggest the symbolic candicates of a numeric spline-based
            activation function
        lock():
            lock activation functions to share parameters
        unlock():
            unlock locked activations
        get_range():
            get the input and output ranges of an activation function
        plot():
            plot the diagram of KAN
        train_kan():
            train KAN
        prune():
            prune KAN
        remove_edge():
            remove some edge of KAN
        remove_node():
            remove some node of KAN
        auto_symbolic():
            automatically fit all splines to be symbolic functions
        symbolic_formula():
            obtain the symbolic formula of the KAN network
    """

    def __init__(
        self,
        width=None,
        grid=3,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        base_fun=torch.nn.SiLU(),
        symbolic_enabled=True,
        bias_trainable=True,
        grid_eps=1.0,
        grid_range=[-1, 1],
        sp_trainable=True,
        sb_trainable=True,
        seed=0,
    ):
        """
        initalize a KAN model

        Args:
        -----
            width : list of int
                :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons
                in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            noise_scale : float
                initial injected noise to spline. Default: 0.1.
            base_fun : fun
                the residual function b(x). Default: torch.nn.SiLU().
            symbolic_enabled : bool
                compute or skip symbolic computations (for efficiency).
                By default: True.
            bias_trainable : bool
                bias parameters are updated or not. By default: True
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1,
                the grid is partitioned using percentiles of samples.
                0 < grid_eps < 1 interpolates between the two extremes.
                Default: 0.02.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            seed : int
                random seed

        Returns:
        --------
            self

        Example
        -------
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> (model.act_fun[0].in_dim, model.act_fun[0].out_dim), \
            (model.act_fun[1].in_dim, model.act_fun[1].out_dim)
        ((2, 5), (5, 1))
        """
        super(KAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # initializeing the numerical front ###

        self.biases = []
        self.act_fun = []
        self.depth = len(width) - 1
        self.width = width

        for l in range(self.depth):
            # splines
            scale_base = (
                1 / np.sqrt(width[l])
                + (
                    torch.randn(
                        width[l] * width[l + 1],
                    )
                    * 2
                    - 1
                )
                * noise_scale_base
            )
            sp_batch = KANLayer(
                in_dim=width[l],
                out_dim=width[l + 1],
                num=grid,
                k=k,
                noise_scale=noise_scale,
                scale_base=scale_base,
                scale_sp=1.0,
                base_fun=base_fun,
                grid_eps=grid_eps,
                grid_range=grid_range,
                sp_trainable=sp_trainable,
                sb_trainable=sb_trainable,
            )
            self.act_fun.append(sp_batch)

            # bias
            bias = nn.Linear(width[l + 1], 1, bias=False).requires_grad_(
                bias_trainable
            )
            bias.weight.data *= 0.0
            self.biases.append(bias)

        self.biases = nn.ModuleList(self.biases)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        # initializing the symbolic front ###
        self.symbolic_fun = []
        for l in range(self.depth):
            sb_batch = Symbolic_KANLayer(in_dim=width[l], out_dim=width[l + 1])
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled

    def forward(self, x):
        """
        KAN forward

        Args:
        -----
            x : 2D torch.float
                inputs, shape (batch, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (batch, output dimension)

        Example
        -------
        >>> model = KAN(width=[2,5,3], grid=5, k=3)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x).shape
        torch.Size([100, 3])
        """

        shape_size = len(x.shape)

        if shape_size == 3:
            B, C, T = x.shape
        elif shape_size == 2:
            B, T = x.shape
        else:
            raise NotImplementedError()

        x = x.view(-1, T)

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = (
                self.act_fun[l](x)
            )

            if self.symbolic_enabled:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x)
            else:
                x_symbolic = 0.0
                postacts_symbolic = 0.0

            x = x_numerical + x_symbolic
            postacts = postacts_numerical + postacts_symbolic

            # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
            grid_reshape = self.act_fun[l].grid.reshape(
                self.width[l + 1], self.width[l], -1
            )
            input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range)
            self.acts_scale_std.append(torch.std(postacts, dim=0))
            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight
            self.acts.append(x)

        U = x.shape[1]

        if shape_size == 3:
            x = x.view(B, C, U)
        elif shape_size == 2:
            assert x.shape == (B, U)

        return x

    def train_kan(
        self,
        dataset,
        opt="LBFGS",
        steps=100,
        log=1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        update_grid=True,
        grid_update_num=10,
        loss_fn=None,
        lr=1.0,
        stop_grid_update_step=50,
        batch=-1,
        small_mag_threshold=1e-16,
        small_reg_factor=1.0,
        metrics=None,
        sglr_avoid=False,
        save_fig=False,
        in_vars=None,
        out_vars=None,
        beta=3,
        save_fig_freq=1,
        img_folder="./video",
        device="cpu",
    ):
        """
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'],
                dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            batch : int
                batch size, if -1 then full.
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to
                apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large
                factos
            device : str
                device
            save_fig_freq : int
                save figure every (save_fig_freq) step

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization

        Example
        -------
        >>> # for interactive examples, please see demos
        >>> from utils import create_dataset
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> model.plot()
        """

        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (
                    x + (factor - 1) * th
                )

            reg_ = 0.0
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(
                    -1,
                )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = -torch.sum(p * torch.log2(p + 1e-4))
                reg_ += (
                    lamb_l1 * l1 + lamb_entropy * entropy
                )  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(
                    torch.mean(torch.abs(self.act_fun[i].coef), dim=1)
                )
                coeff_diff_l1 = torch.sum(
                    torch.mean(
                        torch.abs(torch.diff(self.act_fun[i].coef)), dim=1
                    )
                )
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc="description", ncols=100)

        if loss_fn is None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(
                self.parameters(),
                lr=lr,
                history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32,
                tolerance_change=1e-32,
                tolerance_ys=1e-32,
            )

        results = {}
        results["train_loss"] = []
        results["test_loss"] = []
        results["reg"] = []
        if metrics is not None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size = dataset["train_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset["train_input"][train_id].to(device))
            if sglr_avoid:
                id_ = torch.where(~torch.isnan(torch.sum(pred, dim=1)))[0]
                train_loss = loss_fn(
                    pred[id_], dataset["train_label"][train_id][id_].to(device)
                )
            else:
                train_loss = loss_fn(
                    pred, dataset["train_label"][train_id].to(device)
                )
            reg_ = reg(self.acts_scale)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:

            train_id = np.random.choice(
                dataset["train_input"].shape[0], batch_size, replace=False
            )
            test_id = np.random.choice(
                dataset["test_input"].shape[0], batch_size_test, replace=False
            )

            if (
                _ % grid_update_freq == 0
                and _ < stop_grid_update_step
                and update_grid
            ):
                self.update_grid_from_samples(
                    dataset["train_input"][train_id].to(device)
                )

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(
                    dataset["train_input"][train_id].to(device)
                )
                if sglr_avoid:
                    id_ = torch.where(~torch.isnan(torch.sum(pred, dim=1)))[0]
                    train_loss = loss_fn(
                        pred[id_],
                        dataset["train_label"][train_id][id_].to(device),
                    )
                else:
                    train_loss = loss_fn(
                        pred, dataset["train_label"][train_id].to(device)
                    )
                reg_ = reg(self.acts_scale)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(
                self.forward(dataset["test_input"][test_id].to(device)),
                dataset["test_label"][test_id].to(device),
            )

            if _ % log == 0:
                pbar.set_description(
                    "train loss: %.2e | test loss: %.2e | reg: %.2e "
                    % (
                        torch.sqrt(train_loss).cpu().detach().numpy(),
                        torch.sqrt(test_loss).cpu().detach().numpy(),
                        reg_.cpu().detach().numpy(),
                    )
                )

            if metrics is not None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results["train_loss"].append(
                torch.sqrt(train_loss).cpu().detach().numpy()
            )
            results["test_loss"].append(
                torch.sqrt(test_loss).cpu().detach().numpy()
            )
            results["reg"].append(reg_.cpu().detach().numpy())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(
                    folder=img_folder,
                    in_vars=in_vars,
                    out_vars=out_vars,
                    title="Step {}".format(_),
                    beta=beta,
                )
                plt.savefig(
                    img_folder + "/" + str(_) + ".jpg",
                    bbox_inches="tight",
                    dpi=200,
                )
                plt.close()

        return results
