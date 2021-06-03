import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal, uniform

class Generator(nn.Module):
    def __init__(self, attr_dim):
        super().__init__()

        self.fc1 = nn.Linear(attr_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)

    def forward(self, noise, atts):
        x = torch.cat((noise, atts), 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


def load_generator(path, attr_number=85, latent_dim=128):
    model = Generator(latent_dim + attr_number).cuda()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


z_dist = normal.Normal(0, 1)
# TODO: make configurable
nz = 128

def generate_batch(netG, batch_size, test_attributes):
    netG.eval()

    # NOTES:
    #  - they translate labels - maybe that's the reason why the sim matrix was weird?
    #  - should we then here operate on translated labels?
    # TODO: randomize labels and return data of exactly batch_size

    num_test_classes = test_attributes.shape[0]
    synth_y = torch.randint(0, num_test_classes, (batch_size,)).cuda()
    synth_attr = torch.index_select(torch.FloatTensor(test_attributes).cuda(), 0, synth_y)
    noise = z_dist.sample((batch_size, nz)).cuda()

    synth_X = netG(noise, synth_attr).detach()

    assert len(synth_y.shape) == 1
    assert synth_X.shape[0] == synth_y.shape[0]

    return synth_X, synth_y
