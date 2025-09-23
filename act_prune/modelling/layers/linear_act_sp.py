import torch
from torch import nn

# from fast_hadamard_transform import hadamard_transform

COLLECT_ETA_PRINT_COUNT = 0
class Linear_act_sp(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        sparsity_type=None,  # [None, "semi-structured_act_magnitude", "unstructured_act_magnitude"]
        transformation_type=None,
        sparsity_ratio=None,  # if sparsity_type is "unstructured_act_magnitude"
        prune_n=None,  # if sparsity_type is "semi-structured_act_magnitude"
        prune_m=None,  # if sparsity_type is "semi-structured_act_magnitude"
        name=None,
        additional_transformation=None,
        eta_buffer_size=100,  
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.transformation_type = transformation_type
        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.register_buffer("weight", None)
        self.name = name
        self.additional_transformation = additional_transformation
        self.eta_buffer_size = eta_buffer_size
        self.eta_collection_completed = False

        self.register_buffer("eta_buffer", torch.zeros(eta_buffer_size))
        self.register_buffer("eta_counter", torch.tensor(0))
        self.register_buffer("static_eta", torch.tensor(float('nan')))

        if sparsity_type == "semi-structured_act_grad_acc":
            self.grad_input = None

        if self.transformation_type == "learnable":
            v = torch.zeros((1, out_features))
            self.shift = nn.Parameter(v)

        if self.sparsity_type == "semi-structured_act_magnitude_var_weight":
            self.var_weight = None

    def unstructured_magnitude_pruner(self, x, sparsity_ratio):
        orig_shape = x.shape
        num_elements_to_keep = int(orig_shape[1] * (1.0 - sparsity_ratio))

        _, idx = torch.topk(x.abs(), num_elements_to_keep, dim=1, sorted=False)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=1, index=idx, value=True)
        x_sp = x * mask
        return x_sp

    def semi_structural_magnitude_pruner(self, x, prune_n=2, prune_m=4):
        orig_shape = x.shape
        x_1d = x.view(-1, prune_m)

        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        return x_sp
    
    def semi_structural_magnitude_columnwise_pruner(self, x, prune_n=2, prune_m=4):
        orig_shape = x.shape
        x_1d = x.view(-1, prune_m)

        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        return x_sp

    def semi_structural_act_grad_acc(self, x, prune_n=2, prune_m=4):
        orig_shape = x.shape

        grad_input = self.grad_input.view(-1, orig_shape[-1])

        x_1d = (x * grad_input).view(-1, prune_m)
        
        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        return x_sp

    def semi_structural_clact_pruner(self, x, prune_n=2, prune_m=4):
        """
        If we remove X_{it} from the input activation X:
            L_{cos_{ti}} = |X_it| * sqrt(sum_p X_pt^2)  / sqrt(sum_j X_ij^2)
        h is hidden dimension, l is sequence length.
        """
    
        abs_x = torch.abs(x) # |X_it|
        denominator = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))  # sqrt(sum_j X_ij^2)
        col_norms = torch.sqrt(torch.sum(x ** 2, dim=0, keepdim=True))  # sqrt(sum_p X_pt^2)
    
        L_metric = abs_x / (denominator + 1e-8) * col_norms

        orig_shape = L_metric.shape
        L_metric_1d = L_metric.view(-1, prune_m)
    
        _, idx = torch.topk(L_metric_1d, prune_n, dim=1, largest=False, sorted=False)
        mask_1d = torch.ones_like(L_metric_1d, dtype=torch.bool)
        mask_1d.scatter_(dim=1, index=idx, value=False)
        mask = mask_1d.view(orig_shape).view_as(x)
        x_sp = x * mask
        return x_sp

    def semi_structural_magnitude_var_weight_pruner(self, x, prune_n=2, prune_m=4):
        orig_shape = x.shape
        x_1d = (x * self.var_weight).view(-1, prune_m)

        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        return x_sp

    
    def semi_structural_amber_pruner(self, x, prune_n=2, prune_m=16):
        
        weights = self.weight
        x_flat = x.view(-1, weights.shape[1])  # [N, in_features]
    
        # Outlier Removal
        flat_weights = weights.view(-1)
        if flat_weights.numel() > 100000:
            sample_size = min(50000, flat_weights.numel())
            indices = torch.randint(0, flat_weights.numel(), (sample_size,), device=flat_weights.device)
            sample = flat_weights[indices].float()
        else:
            sample = flat_weights.float()
    
        q0_005 = torch.quantile(sample, 0.005)
        q0_995 = torch.quantile(sample, 0.995)
        
        mask_weights = (weights >= q0_005) & (weights <= q0_995)
        if not mask_weights.any():
            mask_weights = torch.ones_like(weights, dtype=torch.bool)
    
        filtered_weights = weights[mask_weights]
    
        # Normalization
        mean_w = filtered_weights.mean()
        std_w = filtered_weights.std(unbiased=False)
        std_w = torch.clamp(std_w, min=1e-8)
        normalized_weights = (weights - mean_w) / std_w
    
        # Channel-wise Scoring
        channel_l2_norms = torch.norm(normalized_weights, dim=0)
    
        # Min norm among all channels
        min_channel_norm = channel_l2_norms.min()
        min_channel_norm = torch.clamp(min_channel_norm, min=1e-8)
    
        # f(Ŵ_{:,j}) = ||Ŵ_{:,j}||_2 / min_k ||Ŵ_{:,k}||_2
        f_values = channel_l2_norms / min_channel_norm  # [in_features]
    
        # S_ij = |X_ij| * f(Ŵ_{:,j})
        scores = torch.abs(x_flat) * f_values.unsqueeze(0)  # [N, in_features]
    
        scores_2d = scores.view(-1, prune_m)  # [N*in_features/prune_m, prune_m]
        _, topk_indices = torch.topk(scores_2d, k=prune_n, dim=1, largest=True, sorted=False)
    
        mask_2d = torch.zeros_like(scores_2d, dtype=torch.bool)
        mask_2d.scatter_(1, topk_indices, True)
        mask_flat = mask_2d.view_as(x_flat)  # [N, in_features]
    
        x_pruned = x_flat * mask_flat
    
        return x_pruned.view_as(x)
        

    def variance_factor(self, x, x_sp):
        var_ratio = torch.var(x, dim=1, keepdim=True) / torch.clamp(torch.var(x_sp, dim=1, keepdim=True), min=1e-9)
        v = torch.sqrt(var_ratio)
        return v
    
    def variance_transformation(self, x, x_sp):
        v = self.variance_factor(x, x_sp)
        corr_x_sp = v * x_sp
        return corr_x_sp

    def bias_term(self, x):
        eta = torch.median(x, dim=1, keepdim=True)[0]
        return eta
        

    def collect_eta(self, x):
        global COLLECT_ETA_PRINT_COUNT
        
        current_eta = self.bias_term(x).mean()
        
        if self.eta_counter < self.eta_buffer_size:
            self.eta_buffer[self.eta_counter] = current_eta
            self.eta_counter += 1
            
            if self.eta_counter == self.eta_buffer_size:
                self.static_eta = self.eta_buffer.mean()
                if COLLECT_ETA_PRINT_COUNT < 1:
                    print(f"Static eta collected: {self.static_eta.item()}")
                    COLLECT_ETA_PRINT_COUNT += 1

    def get_eta_to_use(self, x):
        if not torch.isnan(self.static_eta):
            return self.static_eta.to(x.device, dtype=x.dtype).expand(x.size(0), 1)
        else:
            return self.bias_term(x)

    def shift_transformation(self, x, pruner, eta):
        if eta.device != x.device or eta.dtype != x.dtype:
            eta = eta.to(x.device, dtype=x.dtype)
        x_shifted = x - eta
        x_sp = pruner(x_shifted)
        x_sp_shifted = x_sp + eta
        return x_sp_shifted

    def scaling_transformation(self, x, pruner):
        max_act = torch.max(torch.abs(x), dim=0).values
        max_weight = torch.max(torch.abs(self.weight), dim=0).values
        s = torch.sqrt(max_act / max_weight.clamp(min=1e-8))
        x_flat_sp = pruner(x / s)
        scaled_weight = self.weight * s.unsqueeze(0)
        return x_flat_sp @ scaled_weight.t()
        
    def learnable_transformation(self, x, pruner):
        x_sp = pruner(x)

        # eta = self.bias_term(x)
        # x_shifted = x - eta
        # x_sp = pruner(x_shifted)
        # v = self.variance_factor(x_shifted, x_sp)
        # x_sp_shifted = v * x_sp + eta

        
        # eta = self.local_bias_term(x)
        # x_sp_shifted = self.shift_transformation(x, pruner, eta)


        # if not hasattr(self, 'eta'):
        #     self.eta = nn.Parameter(self.bias_term(x))
            
        # bs = x.shape[0]
        # x_sp_shifted = self.shift_transformation(x, pruner, self.eta[:bs])

        # if not hasattr(self, 'v'):
        #     x_sp = pruner(x)
        #     self.v = nn.Parameter(self.variance_factor(x, x_sp))
        
        # corr_x_sp_shifted = self.v * x_sp_shifted
        
        return x_sp

    def prune_with_additional_transformation(self, x, pruner):
        if self.additional_transformation == "scaling":
            return self.scaling_transformation(x, pruner)
        return pruner(x) @ self.weight.t()

    def add_grad(self, grad_input, grad_output):
        if self.grad_input is None:
            self.grad_input = grad_input[0]
        else:
            self.grad_input += grad_input[0]

    def forward(self, x):
        bs, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)
        out = None

        if self.transformation_type == "shift" and torch.isnan(self.static_eta):
            self.collect_eta(x_flat)

        # Without pruning
        if self.sparsity_type is None:
            out = x @ self.weight.t()

        # Semi-structured with transformation logic
        elif self.sparsity_type in ["semi-structured_act_magnitude", "semi-structured_act_magnitude_var_weight", "semi-structured_act_grad_acc"]:
            if self.sparsity_type == "semi-structured_act_magnitude":
                pruner = lambda z: self.semi_structural_magnitude_pruner(z, self.prune_n, self.prune_m)
            
            elif self.sparsity_type == "semi-structured_act_grad_acc":
                pruner = lambda z: self.semi_structural_act_grad_acc(z, self.prune_n, self.prune_m)

            elif self.sparsity_type == "semi-structured_act_magnitude_var_weight":
                pruner = lambda z: self.semi_structural_magnitude_var_weight_pruner(z, self.prune_n, self.prune_m)

            if self.transformation_type == "variance":
                x_sp = pruner(x_flat) 
                out = self.variance_transformation(x_flat, x_sp) @ self.weight.t()

            if self.transformation_type == "shift":
                eta_to_use = self.get_eta_to_use(x_flat)
                out = self.shift_transformation(x_flat, pruner, eta_to_use) @ self.weight.t()
            
            elif self.transformation_type == "learnable":
                x_sp = self.learnable_transformation(x_flat, pruner)
                out = torch.matmul(x_sp, self.weight.t()) + self.shift
                # out = self.v * out
                # x_sp = x_sp.to_dense()
            elif self.transformation_type == "scaling" or self.additional_transformation == "scaling":
                out = self.scaling_transformation(x_flat, pruner)
            else:
                out = pruner(x_flat) @ self.weight.t()
        
        # Unstructured pruning
        elif self.sparsity_type == "unstructured_act_magnitude":
            pruner = lambda z: self.unstructured_magnitude_pruner(z, self.sparsity_ratio)
            out = self.prune_with_additional_transformation(x_flat, pruner)

        # L-based pruning from shirin-shift-transform
        elif self.sparsity_type == "semi_structural_clact":
            x_sp = self.semi_structural_clact_pruner(x_flat, self.prune_n, self.prune_m)
            out = x_sp @ self.weight.t()

        elif self.sparsity_type == "semi_structural_amber":
            pruner = lambda z: self.semi_structural_amber_pruner(z, self.prune_n, self.prune_m)
            out = pruner(x_flat) @ self.weight.t()

        else:
            raise ValueError(f"Unknown sparsity_type: {self.sparsity_type}")

        return out.view(bs, seq_len, -1)

    @classmethod
    def from_original(
        cls,
        orig_linear,
        sparsity_type=None,
        sparsity_ratio=None,
        transformation_type=None,
        prune_n=None,
        prune_m=None,
        name=None,
        additional_transformation=None,
    ):
        linear_sp = cls(
            orig_linear.in_features,
            orig_linear.out_features,
            sparsity_type=sparsity_type,
            transformation_type=transformation_type,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            name=name,
            additional_transformation=additional_transformation,
        )

        linear_sp.weight = orig_linear.weight.data

        if transformation_type == "learnable":
            linear_sp.shift.data = linear_sp.shift.data.to(
                dtype=orig_linear.weight.dtype,
                # dtype=torch.bfloat16,
                device=orig_linear.weight.device
            )
        
        if sparsity_type == "semi-structured_act_magnitude_var_weight":
            linear_sp.var_weight = torch.var(orig_linear.weight.data, dim=0, keepdim=True)

        return linear_sp
