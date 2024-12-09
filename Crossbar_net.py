import torch
import torch.nn as nn
import torch.nn.functional as F
from mapping import weight_mapping, output_unmapping, quantization
from CrossbarModels.Crossbar_Models_pytorch import IdealModel
from plot_utils import plot_data

# Custom Layer implementation
class CustomLayer(nn.Module):
    def __init__(self, weight, bias, parasiticResistance, R_hrs, R_lrs, model_function, device, debug=False, bits=0, correction=False):
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
        self.currents_corrected = None
        self.correction = correction


    def forward(self, x):

        if self.model_function == IdealModel and not self.debug:
            return torch.matmul(x.unsqueeze(0) if len(x.shape) == 1 else x, self.weight)+ self.bias
        
        if self.bits and self.model_function != IdealModel:
            conductances = quantization(weight_mapping(self.weight, self.R_hrs, self.R_lrs), self.R_lrs, self.R_hrs, self.bits)
        else:
            conductances = weight_mapping(self.weight, self.R_hrs, self.R_lrs)

        Currents = self.model_function(conductances,x,self.parasiticResistance)

        if self.debug:
            self.currents = Currents.detach().cpu()

        if self.correction and self.model_function != IdealModel:
            IdealCurrents = IdealModel(conductances,x,self.parasiticResistance)
            correction_coefficient = (torch.max(IdealCurrents, dim=1).values - torch.min(IdealCurrents, dim=1).values) / (torch.max(Currents, dim=1).values - torch.min(Currents, dim=1).values + 1e-8)
            Currents = (Currents-(torch.mean(Currents,1).unsqueeze(1))) * correction_coefficient.unsqueeze(1) + (torch.mean(IdealCurrents,1)).unsqueeze(1)

        if self.debug:
            self.currents_corrected = Currents.detach().cpu()

        output = output_unmapping(self.weight.T, Currents, x, self.R_hrs, self.R_lrs)

        return output + self.bias
        

# Custom Neural Network
class CustomNet(nn.Module):
    def __init__(self, weights, parasiticResistance, R_hrs, R_lrs, model_function, device, debug=False, bits=0, correction=False):
        super(CustomNet, self).__init__()
        self.fc1 = CustomLayer(weights['fc1_weights'], weights['fc1_bias'], parasiticResistance, R_hrs, R_lrs, model_function, device, debug, bits, correction)
        self.fc2 = CustomLayer(weights['fc2_weights'], weights['fc2_bias'], parasiticResistance, R_hrs, R_lrs, model_function, device, debug, bits, correction)

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
