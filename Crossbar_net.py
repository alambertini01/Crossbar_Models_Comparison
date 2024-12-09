import torch
import torch.nn as nn
import torch.nn.functional as F
from mapping import weight_mapping, output_unmapping, quantization
from CrossbarModels.Crossbar_Models_pytorch import IdealModel
from plot_utils import plot_data


class CustomLayer(nn.Module):
    def __init__(self, weight, bias, parasiticResistance, R_hrs, R_lrs, model_function, device, 
                 debug=False, bits=0, correction=False, max_array_size=784):
        super(CustomLayer, self).__init__()
        self.device = device
        self.weight = nn.Parameter(weight.clone().detach().to(self.device))  # (input_size, output_size)
        self.bias = nn.Parameter(bias.clone().detach().to(self.device))      # (output_size)
        self.R_hrs = R_hrs
        self.R_lrs = R_lrs
        self.bits = bits
        self.parasiticResistance = parasiticResistance
        self.model_function = model_function
        self.debug = debug
        self.correction = correction
        self.max_array_size = max_array_size
        self.currents = None
        self.currents_corrected = None

    def process_tile(self, tile_weight, tile_x):
        # tile_weight: (in_tile, out_tile), tile_x: (batch_size, in_tile)
        # Compute conductances
        if self.bits and self.model_function != IdealModel:
            cond = quantization(weight_mapping(tile_weight, self.R_hrs, self.R_lrs), 
                                self.R_lrs, self.R_hrs, self.bits)
        else:
            cond = weight_mapping(tile_weight, self.R_hrs, self.R_lrs)

        # Compute currents
        Currents = self.model_function(cond, tile_x, self.parasiticResistance)

        Currents_before_corr = Currents.detach().cpu()

        # Apply correction if needed
        if self.correction and self.model_function != IdealModel:
            IdealCurrents = IdealModel(cond, tile_x, self.parasiticResistance)
            coeff = (torch.max(IdealCurrents, dim=1).values - torch.min(IdealCurrents, dim=1).values) / \
                    (torch.max(Currents, dim=1).values - torch.min(Currents, dim=1).values + 1e-8)
            Currents = (Currents - Currents.mean(dim=1, keepdim=True)) * coeff.unsqueeze(1) + IdealCurrents.mean(dim=1, keepdim=True)

        Currents_after_corr = Currents.detach().cpu()

        # Compute partial output
        partial_out = output_unmapping(tile_weight.T, Currents, tile_x, self.R_hrs, self.R_lrs)

        return partial_out, Currents_before_corr, Currents_after_corr

    def forward(self, x):
        # x: (batch_size, input_size)
        batch_size, in_features = x.shape
        input_size, output_size = self.weight.shape

        # One-shot if possible with IdealModel and no debug
        if self.model_function == IdealModel and not self.debug and \
           self.max_array_size >= in_features and self.max_array_size >= output_size:
            return torch.matmul(x, self.weight) + self.bias

        # Tiling logic
        in_tiles = (in_features + self.max_array_size - 1) // self.max_array_size
        out_tiles = (output_size + self.max_array_size - 1) // self.max_array_size

        output_acc = torch.zeros(batch_size, output_size, device=self.device)

        if self.debug:
            # We'll sum currents from each tile
            full_currents = torch.zeros(batch_size, output_size, device='cpu')
            full_currents_corrected = torch.zeros(batch_size, output_size, device='cpu')

        for i_t in range(in_tiles):
            i_start = i_t * self.max_array_size
            i_end = min(i_start + self.max_array_size, in_features)
            tile_x = x[:, i_start:i_end]

            for o_t in range(out_tiles):
                o_start = o_t * self.max_array_size
                o_end = min(o_start + self.max_array_size, output_size)
                tile_w = self.weight[i_start:i_end, o_start:o_end]

                partial_out, Currents_before, Currents_after = self.process_tile(tile_w, tile_x)
                output_acc[:, o_start:o_end] += partial_out

                if self.debug:
                    full_currents[:, o_start:o_end] += Currents_before
                    full_currents_corrected[:, o_start:o_end] += Currents_after

        if self.debug:
            self.currents = full_currents
            self.currents_corrected = full_currents_corrected

        return output_acc + self.bias


class CustomNet(nn.Module):
    def __init__(self, weights, parasiticResistance, R_hrs, R_lrs, model_function, device, 
                 debug=False, bits=0, correction=False, max_array_size=784):
        super(CustomNet, self).__init__()
        self.fc1 = CustomLayer(weights['fc1_weights'], weights['fc1_bias'], parasiticResistance, 
                               R_hrs, R_lrs, model_function, device, debug, bits, correction, max_array_size)
        self.fc2 = CustomLayer(weights['fc2_weights'], weights['fc2_bias'], parasiticResistance, 
                               R_hrs, R_lrs, model_function, device, debug, bits, correction, max_array_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Evaluate Model
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, all_preds, all_targets
