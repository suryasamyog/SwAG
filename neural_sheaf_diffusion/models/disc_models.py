# ram


import torch
import torch.nn.functional as F
import torch_sparse

from torch import nn
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant
# from utils.preconnection import precompute_connection_laplacian
import matplotlib.pyplot as plt
import seaborn as sns 
from torch_geometric.nn import LayerNorm

# class DiscreteDiagSheafDiffusion(SheafDiffusion):

#     def __init__(self, edge_index, args):
#         super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
#         assert args['d'] > 0
    

#         self.lin_right_weights = nn.ModuleList()
#         self.lin_left_weights = nn.ModuleList()
#         # self.lin_left_weights = nn.ParameterList()

#         # self.batch_norms = nn.ModuleList()
        
#         if self.right_weights:
#             for i in range(self.layers):
#                 self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
#                 nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
#         if self.left_weights:
#             for i in range(self.layers):
#                 self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
#                 nn.init.eye_(self.lin_left_weights[-1].weight.data)

        
#         self.sheaf_learners = nn.ModuleList()

#         num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
#         for i in range(num_sheaf_learners):
#             if self.sparse_learner:
#                 self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
#                     self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
#             else:

#                 self.sheaf_learners.append(LocalConcatSheafLearner(
#                     self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
#         self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
#                                                          normalised=self.normalised,
#                                                          deg_normalised=self.deg_normalised,
#                                                          add_hp=self.add_hp, add_lp=self.add_lp)

#         # self.epsilons = nn.ParameterList()
#         # for i in range(self.layers):
#         #     self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

#         self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
#         if self.second_linear:
#             self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         # self.bn = nn.BatchNorm1d(self.hidden_dim)
#         # self.ln = nn.LayerNorm(args["hidden_channels"])
#         if args["norm"] == "group":
#             print("Using GroupNorm")
#             self.gn = nn.GroupNorm(num_groups=self.final_d, num_channels=self.hidden_dim)
#             self.bn = None
#         elif args["norm"] == "batch":
#             print("Using BatchNorm")
#             self.bn = nn.BatchNorm1d(self.hidden_dim, momentum=0.01)
#             self.gn = None
#         else:
#             print("Using no norm")
#             self.bn = None
#             self.gn = None
#         if self.use_act:
#             self.act = nn.ELU()

#         # self.act = nn.PReLU()
#         # self.act = nn.PReLU()
#         # self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
#         # self.reset_parameters()

#     def reset_parameters(self):
#         self.lin1.reset_parameters()

#     def forward(self, x):
#         x = F.dropout(x, p=self.input_dropout, training=self.training)
#         # print(x.shape)
#         x = self.lin1(x)
#         if self.gn:
#             x = self.gn(x)
#         if self.bn:
#             x = self.bn(x)
#         if self.use_act:
#             # x = F.elu(x)
#             # x = F.tanh(x)
#             x = self.act(x)
#         # if self.bn:
#         #     x = self.bn(x)
#         # x = F.dropout(x, p=self.dropout, training=self.training)
#         if self.second_linear:
#             x = self.lin12(x)
#         x = x.view(self.graph_size * self.final_d, -1)
#         x0 = x
#         for layer in range(self.layers):
#             if layer == 0 or self.nonlinear:
#                 x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
#                 maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
#                 L, trans_maps = self.laplacian_builder(maps)                
#                 self.sheaf_learners[layer].set_L(trans_maps)

            
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             x_int = 0
#             if self.left_weights:
#                 x = x.t().reshape(-1, self.final_d)
#                 x = self.lin_left_weights[layer](x)
#                 # x_int = x
#                 x = x.reshape(-1, self.graph_size * self.final_d).t()

                
#             if self.right_weights:
#                 x = self.lin_right_weights[layer](x)

            
#             x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)


#             if self.use_act:
#                 # x = F.elu(x)
#                 # x = F.tanh(x)
#                 x = self.act(x)
            
#             # x = self.ln(x)
            
#             # coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
#             # x0 = coeff * x0 - x
#             x0 = x0 - x
#             x = x0

#         # energy = torch.sum(x * torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x))
#         # print(energy/(torch.sum(x * x) + 1e-8))
#         x = x.reshape(self.graph_size, -1)
#         return {"z": x, "maps": maps} 
#         # return x
#         # x = self.lin2(x)
#         # return x, trans_maps
#         # return x, maps
#         # return x
#         # return x, x_int

#         # x = x.reshape(self.graph_size, -1)
#         # x = self.lin2(x)
#         # return x, trans_maps
#         # x = x.reshape(self.graph_size, -1)
#         # x = self.lin2(x)
#         # return F.log_softmax(x, dim=1)

class DiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0
    

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        # self.lin_left_weights = nn.ParameterList()

        # self.batch_norms = nn.ModuleList()
        
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        
        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:

                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        # self.epsilons = nn.ParameterList()
        # for i in range(self.layers):
        #     self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.bn = nn.BatchNorm1d(self.hidden_dim)
        # self.ln = nn.LayerNorm(args["hidden_channels"])
        if args["norm"] == "group":
            print("Using GroupNorm")
            self.gn = nn.GroupNorm(num_groups=self.final_d, num_channels=self.hidden_dim)
            self.bn = None
        elif args["norm"] == "batch":
            print("Using BatchNorm")
            self.bn = nn.BatchNorm1d(self.hidden_dim, momentum=0.01)
            self.gn = None
        else:
            print("Using no norm")
            self.bn = None
            self.gn = None
        if self.use_act:
            self.act = nn.ELU()

        # self.act = nn.PReLU()
        # self.act = nn.PReLU()
        # self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        # self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        # print(x.shape)
        x = self.lin1(x)
        if self.gn:
            x = self.gn(x)
        if self.bn:
            x = self.bn(x)
        if self.use_act:
            # x = F.elu(x)
            # x = F.tanh(x)
            x = self.act(x)
        # if self.bn:
        #     x = self.bn(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)
        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, non_diag_L, diag_L, trans_maps = self.laplacian_builder(maps)                
                self.sheaf_learners[layer].set_L(trans_maps)
                # print(trans_maps.shape)

            
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_int = 0
            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                # x_int = x
                x = x.reshape(-1, self.graph_size * self.final_d).t()

                
            if self.right_weights:
                x = self.lin_right_weights[layer](x)
            
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            # x = x0

            # print(torch.equal(x1+x2, x0))
            # print(x2.shape)
            # raise

            if self.use_act:
                # x = F.elu(x)
                # x = F.tanh(x)
                x = self.act(x)
            
            # x = self.ln(x)
            
            # coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            # x0 = coeff * x0 - x
            x0 = x0 - x
            x = x0
        
        x1 = torch_sparse.spmm(diag_L[0], diag_L[1], x.size(0), x.size(0), x)
        x2 = torch_sparse.spmm(non_diag_L[0], non_diag_L[1], x.size(0), x.size(0), x)

        # energy = torch.sum(x * torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x))
        # print(energy/(torch.sum(x * x) + 1e-8))
        x = x.reshape(self.graph_size, -1)
        x1 = x1.reshape(self.graph_size, -1)
        x2 = x2.reshape(self.graph_size, -1)
        # print(L[0].shape)
        return {"z": x, "maps": maps, "L": L, "view1_laplacian_diag": x1, "view2_laplacian_non_diag": x2}


class DiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised
        print("This is the modified BUndleSHeaf")
     

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()


        # self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                print(self.get_param_size())
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

            
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        # self.epsilons = nn.ParameterList()
        # for i in range(self.layers):
        #     self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))


        # self.lin1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False) ### this gave the best for phi=-1
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.bn = nn.BatchNorm1d(self.hidden_dim)
        # self.ln = nn.LayerNorm(args["hidden_channels"])
        # self.act = nn.PReLU()
        # self.gn = nn.GroupNorm(num_groups=self.final_d, num_channels=self.hidden_dim)
        # self.act = nn.ELU()
        if args["norm"] == "group":
            print("Using GroupNorm")
            self.gn = nn.GroupNorm(num_groups=args["hidden_channels"], num_channels=self.hidden_dim)
            self.bn = None
        elif args["norm"] == "batch":
            print("Using BatchNorm")
            self.bn = nn.BatchNorm1d(self.hidden_dim)
            self.gn = None
        else:
            print("Using no norm")
            self.bn = None
            self.gn = None
        if self.use_act:
            self.act = nn.ELU()
        # self.act = nn.Tanh()
        # sefl.act = nn.ReLU()
        # self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
    #     self.reset_parameters()
       

    # def reset_parameters(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             module.reset_parameters()

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def visualize(self, maps, epoch):
        flat_maps = maps.reshape(maps.shape[0] , -1)
        plt.figure(figsize=(12, 10))
        sns.heatmap(flat_maps.T.detach().cpu().numpy(), cmap="coolwarm", cbar=True)
        plt.savefig(f"heatmaps/epoch_{epoch}_heatmap.png")

    def forward(self, x):

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.gn:
            x = self.gn(x)
        if self.use_act:
            # x = F.elu(x)
            # x = F.tanh(x)
            x = self.act(x)
        if self.bn:
            x = self.bn(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
 
        if self.second_linear:
            x = self.lin12(x)
        
        x = x.view(self.graph_size * self.final_d, -1)

      
        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
           
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)

                # print("Learning maps")
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None

                # print("Learning edge weights")
                # L, trans_maps, orth_maps = self.laplacian_builder(maps, edge_weights)
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                # if epoch % 250 == 0 and online:
                #     self.visualize(trans_maps, epoch)
              
                self.sheaf_learners[layer].set_L(trans_maps)


            x = F.dropout(x, p=self.dropout, training=self.training)

            # x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

        
            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

         
            if self.use_act:
                # x = F.elu(x)
                # x = F.tanh(x)
                x = self.act(x)

            # x = self.ln(x)
            # x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x0 = x0 - x
            x = x0
            # print(x.shape)
            # raise

        x = x.reshape(self.graph_size, -1)
        return x, trans_maps
        # return x

        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=1)
        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return x, trans_maps 


#######################
##Precomputed Version

class DiscretePreBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, data, args):
        super(DiscretePreBundleSheafDiffusion, self).__init__(data.edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        x, edge_index = data.x, data.edge_index
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        # self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        # self.sheaf_learners = nn.ModuleList()
        # self.weight_learners = nn.ModuleList()

        # num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        # for i in range(num_sheaf_learners):
        #     if self.sparse_learner:
        #         self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
        #             self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
        #     else:
        #         self.sheaf_learners.append(LocalConcatSheafLearner(
        #             self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

            
        #     if self.use_edge_weights:
        #         self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        # self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
        #     self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
        #     add_lp=self.add_lp, orth_map=self.orth_trans)
        self.L = precompute_connection_laplacian(x, edge_index, self.d, args["graph_size"])

        # self.epsilons = nn.ParameterList()
        # for i in range(self.layers):
        #     self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.gn = nn.GroupNorm(num_groups=self.final_d, num_channels=self.hidden_dim)
        self.act = nn.ELU()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x= self.gn(x)
        if self.use_act:
            # x = F.elu(x)
            x = self.act(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
     
        x = x.view(self.graph_size * self.final_d, -1)
      


        x0 = x
        L, trans_maps, orth_maps = self.L
        L = L.to(self.device)
        for layer in range(self.layers):
            # if layer == 0 or self.nonlinear:
                # x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                # x_maps = x_maps.reshape(self.graph_size, -1)
           
                # maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                # print("Learning maps")
                # edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None

                # print("Learning edge weights")
                # L, trans_maps = self.laplacian_builder(maps, edge_weights)
                # print(L[0].shape, L[1].shape, self.edge_index.shape)
                # self.sheaf_learners[layer].set_L(trans_maps)

            # x = F.dropout(x, p=self.dropout, training=self.training)
            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)
            # x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            # x = self.left_right_linear(x, None, self.lin_right_weights[layer])
            
            # print("In the precompute version")
            # print(x.shape, L[0].shape, L[1].shape)
            # raise

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
        

            if self.use_act:
                x = self.act(x)
                # x = F.elu(x)

            # x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x0 = x0 - x
            x = x0
        return x, orth_maps
        # return x

        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=1)
        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return x, orth_maps

#######################


class DiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))
        return x, None
        # return x
        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=1)