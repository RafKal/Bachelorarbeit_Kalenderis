import torch
import torch.nn as nn

# gradient penalty from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/utils.py

def gradient_penalty(critic, real, fake, device="cuda"):
    BATCH_SIZE, C, L = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, L).to(device)
    interpolated_samples = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_samples)

    # Take the gradient of the scores with respect to the samples
    gradient = torch.autograd.grad(
        inputs=interpolated_samples,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def init_weights(self):
        for module in self.modules():  
            classname = module.__class__.__name__
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                print(classname)
                nn.init.orthogonal_(module.weight.data)
                #nn.init.normal_(module.weight.data, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
                print(classname)
                nn.init.orthogonal_(module.weight.data)
                #nn.init.normal_(module.weight.data, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
              
            elif isinstance(module, nn.LSTM):
                 for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        print(module)
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)
                        # Initialize forget gate bias to 1 (common practice)
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.normal_(module.weight.data, 0, 0.02)
                nn.init.constant_(module.bias.data, 0)         



# costum module getter to assert different learn rates
def get_modules(self):
         # Define different parameter groups
         # Separate parameter lists
            gen_params = []
            critic_params = []

            

            for module in self.critic.modules():  # Only iterate over Critic modules
                if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    critic_params.append({"params": module.parameters(), "lr": 1e-4}) 
                elif isinstance(module, nn.Linear):
                    critic_params.append({"params": module.parameters(), "lr": 1e-4})   
                elif isinstance(module, nn.BatchNorm1d):
                    critic_params.append({"params": module.parameters(), "lr": 1e-4})        
            
            for module in self.gen.modules():  # Only iterate over Generator modules
                if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
                    gen_params.append({"params": module.parameters(), "lr": 1e-4}) 
                elif isinstance(module, nn.LSTM):
                    gen_params.append({"params": module.parameters(), "lr": 1e-4}) 
                elif isinstance(module, nn.Linear):
                    gen_params.append({"params": module.parameters(), "lr": 1e-4}) 
                elif isinstance(module, nn.BatchNorm1d):
                    gen_params.append({"params": module.parameters(), "lr": 1e-4}) 
                     
            return gen_params, critic_params
