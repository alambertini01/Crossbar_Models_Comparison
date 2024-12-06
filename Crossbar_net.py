import torch
import torch.nn as nn
import torch.nn.functional as F
from mapping import weight_mapping, output_unmapping, quantization
from CrossbarModels.Crossbar_Models_pytorch import IdealModel

# Custom Layer implementation
class CustomLayer(nn.Module):
    def __init__(self, weight, bias, parasiticResistance, R_hrs, R_lrs, model_function, device, debug=False, bits=0, bias_correction=False):
        super(CustomLayer, self).__init__()
        self.device = device
        self.weight = nn.Parameter(weight.clone().detach().to(self.device))
        self.bias = nn.Parameter(bias.clone().detach().to(self.device))
        self.R_hrs = R_hrs
        self.R_lrs = R_lrs
        self.bits = bits
        self.parasiticResistance = parasiticResistance
        self.model_function = model_function
        self.debug = debug
        self.currents = None
        self.bias_correction = bias_correction


    def forward(self, x):
        if self.bits and self.model_function != IdealModel:
            conductances = quantization(weight_mapping(self.weight, self.R_hrs, self.R_lrs), self.R_lrs, self.R_hrs, self.bits)
        else:
            conductances = weight_mapping(self.weight, self.R_hrs, self.R_lrs)
        Currents = self.model_function(
            conductances,
            x,
            self.parasiticResistance
        )
        if self.bias_correction:
            IdealCurrents = IdealModel(
                conductances,
                x,
                self.parasiticResistance
            )
            correction = torch.mean(Currents,1)/torch.mean(IdealCurrents,1)
        if self.debug:
            self.currents = Currents.detach().cpu()
        output = output_unmapping(self.weight.T, Currents, x, self.R_hrs, self.R_lrs)
        if self.bias_correction:
            return output + self.bias.repeat(output.shape[0], 1) * correction.unsqueeze(1).repeat(1, self.bias.shape[0])
        else:
            return output + self.bias
        

# Custom Neural Network
class CustomNet(nn.Module):
    def __init__(self, weights, parasiticResistance, R_hrs, R_lrs, model_function, device, debug=False, bits=0, bias_correction=False):
        super(CustomNet, self).__init__()
        self.fc1 = CustomLayer(weights['fc1_weights'], weights['fc1_bias'], parasiticResistance, R_hrs, R_lrs, model_function, device, debug, bits, bias_correction)
        self.fc2 = CustomLayer(weights['fc2_weights'], weights['fc2_bias'], parasiticResistance, R_hrs, R_lrs, model_function, device, debug, bits, bias_correction)

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
