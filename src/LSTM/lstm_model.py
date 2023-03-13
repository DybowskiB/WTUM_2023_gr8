from dataset import LSTMDataset
import pytorch_lightning as pl
from typing import  Tuple
from torch import Tensor, optim, nn
import torch



class LSTMModel(pl.LightningModule):
    def __init__(self, lr=6e-4,
                 input_size=1,
                 output_size=1,
                 hidden_size=10,
                 n_layers=2,
                 set_device = "cpu",
                 window_size=200,
                 batch_size=16) -> None:
        super().__init__()
        
        # Hyperparameters
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.learning_rate = lr
        self.batch_size = batch_size
        
        # Architecture
        self.LSTM = nn.LSTM(input_size,
                            hidden_size,
                            n_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size*window_size, output_size)
        self.loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.device2 = set_device
        
    def forward(self, x) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0) # [50]
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device2) 
        cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device2) 
        hidden = (hidden_state, cell_state) 
        output, (hidden_state, cell_state) = self.LSTM(x, hidden)
        output = self.relu(output)
        output = output.reshape(output.shape[0], -1) 
        output = self.linear(output)
        return output
    
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr = self.learning_rate, betas=(0.9, 0.999), weight_decay=0)
        return optimizer
            
    def training_step(self, batch, batch_idx):
        features, targets = batch
        output = self(features) 
        output = output.view(-1)
        loss = self.loss(output, targets)
        self.log('train_loss', loss, prog_bar=True)
        return {"loss": loss}
        
    def train_dataloader(self):
        traffic_volume_train =  LSTMDataset(train=True)
        train_dataloader = torch.utils.data.DataLoader(traffic_volume_train, batch_size=self.batch_size) 
        return train_dataloader
  
    def validation_step(self, val_batch, batch_idx):
        features, targets = val_batch
        output = self(features) 
        output = output.view(-1)
        loss = self.loss(output, targets)
        self.log('val_loss', loss, prog_bar=True)

    def val_dataloader(self):
        traffic_volume_val =  LSTMDataset(validate=True)
        val_dataloader = torch.utils.data.DataLoader(traffic_volume_val, batch_size=self.batch_size) 
        return val_dataloader