import torch.nn as nn
from config.config_bases import ModelConfigBase
import torch.optim as optim
import torch
import models.utils.pt_training_loops as training_loops
from torch.utils.data import DataLoader
from models.utils.pt_utils import repeat_vector
from models.utils.pt_utils import TimeDistributed
from torchview import draw_graph
import pandas as pd
from models.utils.WaveGAN_utils import gradient_penalty, init_weights, get_modules
import torch.nn.functional as F

# base DCGAN skeleton from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/model.py

class GanConfig(ModelConfigBase):
    def __init__(self):
        self.lr = 1e-4
        self.sequence_length = 512
        self.z_dim = 100   
        self.batch_size = 32 
        self.num_epochs = 20 
        self.model_size = 64
        self.shift_factor = 0 # Phase Shuffle
        
        
        # For W-GAN-GP
        self.critic_iterations = 5 
        self.lambda_GP = 10 

        


class Gan(nn.Module):
    def __init__(self, model_config: GanConfig, device):
        super().__init__()
        self.model_config = model_config
        self.device = device
        self.tracker = GanLossTracker()
        
        
        self.critic = Critic(model_config.n_channels, model_config.batch_size,
                                         model_config.model_size, model_config.print, model_config.shift_factor).to(device)
        init_weights(self.critic)
        
        
        self.gen = Generator(model_config.z_dim, model_config.n_channels,
                              model_config.batch_size , model_config.model_size,  model_config.print ).to(device)
        init_weights(self.gen)
       

        
        gen_params, critic_params = get_modules(self) # costum modules getter for init with different learn rates per module
        self.opt_gen = optim.Adam(gen_params,  betas=[0.5, 0.9])
        self.opt_critic = optim.Adam(critic_params, betas=[0.5, 0.9]) 
         
        self.data_loss = torch.nn.MSELoss()


       



    def test_step(self, real):
        return

    def train_step(self, real):
        
        # PyTorch needs format of [batch size, sequence length, n channels]
        real = torch.permute(real, (0, 2, 1))
        

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(self.model_config.critic_iterations):
            fake =  self.forward() # 1 is dummy input to forego error..

            critic_fake = self.critic(fake).reshape(-1)
            critic_real = self.critic(real).reshape(-1)
            gp = gradient_penalty(self.critic, real, fake, device=self.device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.model_config.lambda_GP * gp
            )

            self.critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = self.critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        self.gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

        mse_loss = self.data_loss(fake, real)
        self.tracker.add_to_train_loss(loss_critic, mse_loss)   

 
    def synthesize_data(self):

        # Reshaping output to be plotted/exported
        data = torch.permute(self.forward().cpu().detach(), (0, 2, 1))
        return data



    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        n_epochs: int,
        gpu_device: str,
        callbacks,
    ):
        return training_loops.fit(
            self, train_loader, test_loader, n_epochs, gpu_device, callbacks
        )
    
    def forward(self, x):
        noise = 2 * (torch.rand(size=(self.model_config.batch_size,  512, 32)).to(self.device)) -1
        fake = self.gen(noise)
        return fake  


class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x # Outputs: output, (h_n, c_n)
       
        return out    

class Generator(nn.Module):
   def __init__(self, data_dim, model_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
 
 
            self._block(model_size*8, model_size*4, 25, 2, 11),  
            self._block(model_size*4, model_size*2, 25, 2, 11), 
            self._block(model_size*2, model_size, 25, 2, 11),  
             nn.ConvTranspose1d(
                model_size, data_dim, 25, 2, 19,
                bias=False, output_padding=1
            ), 
            nn.Tanh(),

            nn.LSTM(data_dim, 256, 1, batch_first=True), 
            GetLSTMOutput(),
            nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True), 
            GetLSTMOutput(),
            nn.LSTM(256, data_dim, 1, batch_first=True), 
            GetLSTMOutput(),
            Swap(0, 2, 1),
            )
        
   def forward(self, x):
        return self.net(x)

   def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
                output_padding=0
            ),
             nn.ReLU(),
        )


class Critic(nn.Module):
   def __init__(self, data_dim, batch_size, model_size, print, shift_factor):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(

            nn.Conv1d(
                data_dim, model_size , 25, 2, 19,
                bias=False
            ), 
            nn.ConstantPad1d((0, 1), 0),
            PhaseShuffle(shift_factor),            
            
            self._block(model_size, model_size*2, 25, 2, 11),  
            PhaseShuffle(shift_factor),  
            
            self._block(model_size*2, model_size*4, 25, 2, 11), 
            PhaseShuffle(shift_factor),  
            
            self._block(model_size*4, model_size*8, 25, 2, 11), 
       
            Reshape([batch_size, -1]),
            nn.Linear(256*model_size, 1),

        )

   def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

   def forward(self, x):
        return self.disc(x)




class GanLossTracker:
    def __init__(self):
        self.init_trackers()

    def get_history(self):
        return {
            "train_loss": self.loss_critic_history, #Critic loss, named train due to bug
            "MSE_loss": self.loss_mse_history,    
        }

    def init_trackers(self):
        self.loss_critic_history = list()
        self.loss_mse_history = list()
        self.reset_epoch_trackers()

    def update_train_loss(self, n_batches):
        self.loss_critic_history.append(self.sum_loss_c / n_batches)
        self.loss_mse_history.append(self.sum_loss_mse / n_batches)

    def add_to_train_loss(self, loss_c, loss_mse):
        self.sum_loss_c += loss_c.item()
        self.sum_loss_mse += loss_mse.item()

    def reset_epoch_trackers(self):
        self.sum_loss_c = 0
        self.sum_loss_mse = 0

    def update_test_loss(self, n_batches):
        return
    
    def get_last_epoch(self):
        return {key: value[-1] for key, value in self.get_history().items()}


# taken from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle


class PhaseRemove(nn.Module):
    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass




# Helper functions
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        

    def forward(self, x):
        return torch.reshape(x, shape=self.shape)
    

class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out

        
class print_output(nn.Module):
    def __init__(self, text, inc_value, print):
        super(print_output, self).__init__()
        self.text = text
        self.inc_value = inc_value
        self.print = print  #True or False

    def forward(self, x): 
        if self.print:
          print(f'{self.text}: {x.shape}')
        return x
    
class print_seperator(nn.Module):
    def __init__(self, text, inc_value, print):
        super(print_seperator, self).__init__()
        self.text = text
        self.inc_value = inc_value
        self.print = print #True or False

    def forward(self, x):
        if self.print:
          print(f'--------------{self.text}---------------')
        return x    
    

class Swap(nn.Module):
    def __init__(self, *dims):
        super(Swap, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


