import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import transforms, utils, datasets
import torchvision.models as models
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torchvision
import os
import gzip
import tarfile
import gc
from PIL import Image
import io
from IPython.core.ultratb import AutoFormattedTB
from copy import deepcopy

# NOTE: This script was drafted in Google colab. If it is deployed 
# elsewhere an different image loading API may be necessary.
from google.colab import files




def upload():
    print('Upload Content Image')
    file_dict = files.upload()
    content_path = io.BytesIO(file_dict[next(iter(file_dict))])

    print('\nUpload Style Image')
    file_dict = files.upload()
    style_path = io.BytesIO(file_dict[next(iter(file_dict))])
    return content_path, style_path

 

def display(tensor, ims, ax):
    """"
    Generate a frame for the animation.
    """
    image = tensor.cpu().clone()  
    image = image.squeeze(0)    # add the batch size dimension  
    image = toPIL(image)
    ax.set_title("Style Transfer")
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(image, animated=True)
    ims.append([im])
    pass



class Normalization(nn.Module):
  def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]).cuda(), std=torch.tensor([0.229, 0.224, 0.225]).cuda()):
      super(Normalization, self).__init__()
      self.mean = torch.tensor(mean).view(-1, 1, 1)
      self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
      return (img - self.mean) / self.std


class VGGIntermediate(nn.Module):
    def __init__(self, requested=[]):
        super(VGGIntermediate, self).__init__()
        self.norm = Normalization().eval()
        self.intermediates = {}
        self.vgg = models.vgg16(pretrained=True).features.eval()
        for i, m in enumerate(self.vgg.children()):
            if isinstance(m, nn.ReLU):   # we want to set the relu layers to NOT do the relu in place. 
                m.inplace = False        # the model has a hard time going backwards on the in place functions. 
            
            if i in requested:
                def curry(i):
                    def hook(module, input, output):
                        self.intermediates[i] = output
                    return hook
                m.register_forward_hook(curry(i))
    
    def forward(self, x):
        self.vgg(self.norm(x))  
        return self.intermediates


def gram_matrix(input_tensor):
  
  # Pull the dimensions
  batch_size, num_channels, height, width = input_tensor.shape
  
  # Generated Gram Matrix (aka a Projection Matrix)
  G = input_tensor.view(num_channels, height*width) #view reshapes tensors
  G_transpose = torch.transpose(G, 0, 1)

  return torch.matmul(G, G_transpose)

  

class ContentLoss(nn.Module):
  def __init__(self, content_layers):
    self.__dict__.update(locals())
    super(ContentLoss, self).__init__()
    self.content_layers = dict()
    for key, value in content_layers.items():
      self.content_layers[key] = value.detach().clone()

  def forward(self, generated_layers, content_layer_nums_list):
    content_loss = 0
    for layer_num in content_layer_nums_list:
      content_loss += 0.5*torch.sum((generated_layers[layer_num] - self.content_layers[layer_num].detach())**2)
    return content_loss

    
    
class StyleLoss(nn.Module):
    def __init__(self, style_layers):
        super(StyleLoss, self).__init__()
        self.__dict__.update(locals())
        self.style_layers = style_layers

    def forward(self, generated_layers, style_layer_nums_list):
        style_loss = 0

        # Assume Equal weighting for each layer.
        # Could add a kwarg for weighting different layers

        for layer_num in style_layer_nums_list:

            # Dimensions
            batch_size, num_channels, height, width = generated_layers[layer_num].shape
            
            # Gram (Projection) Matrices
            G = gram_matrix(generated_layers[layer_num])
            S = gram_matrix(self.style_layers[layer_num].detach())

            # Style Loss
            style_loss += 1/(4*(num_channels**2)*((height*width)**2))*torch.sum((G - S)**2)
        return style_loss



def main():

    # Use this code to upload your own images

    load_and_normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    content_path, style_path = upload()

    print("Content Path: {}".format(content_path))
    print("Style Path: {}".format(style_path))


    # After the images are uploaded on to the local filesystem, you can use:
    content_image_orig = Image.open(content_path)
    content_image = load_and_normalize(np.array(content_image_orig)).unsqueeze(0).cuda()
    style_image_orig = Image.open(style_path)
    style_image = load_and_normalize(np.array(style_image_orig)).unsqueeze(0).cuda()

    # Display the images
    toPIL = transforms.ToPILImage() 

    plt.figure()
    display(style_image, title='Style Image')

    plt.figure()
    display(content_image, title='Content Image')

    vgg_names = ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1", "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2", "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3","maxpool3", "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3","maxpool4", "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3","maxpool5"]

    # Choose the layers to use for style and content transfer
    requested_layer_list = [0, 5, 10, 17, 24]
    # requested_style_layer_list = [0, 5, 10, 17, 24]

    # Create the vgg network in eval mode
    #  with our forward method that returns the outputs of the intermediate layers we requested
    content_model = VGGIntermediate(requested=requested_layer_list).cuda().eval()
    style_model = VGGIntermediate(requested=requested_layer_list).cuda().eval()

    # Cache the outputs of the content and style layers for their respective images
    content_layers = content_model(content_image)
    style_layers = style_model(style_image)

    # Instantiate a content loss module for each content layer 
    #  with the content reference image outputs for that layer for comparison
    content_loss_function = ContentLoss(content_layers)

    # Instantiate a style loss module for each style layer 
    #  with the style reference image outputs for that layer for comparison
    style_loss_function = StyleLoss(style_layers)

    # Initialize pieces for animation
    fig, ax = plt.subplots() #figure()
    ims = []

    # Start with a copy of the content image
    generated_image = content_image.clone().requires_grad_(True)

    #Optimizer
    optimizer = optim.Adam([generated_image], lr=0.1)

    #Total Steps
    total_steps = 151

    #Loss coefficients
    alpha = 0.1 #Content weight
    beta = 10000  #Style weight

    #Which layer(s) to use (options: 0, 5, 10, 17, 24)
    chosen_content_layer_nums = [17, 24]
    chosen_style_layer_nums = [0, 5, 10, 17,24]


    # Run the optimizer on the images to change the image
    #  using the loss of the style and content layers
    #  to backpropagate errors 
    for step in range(total_steps):

        generated_layers = content_model(generated_image)

        #Calculate Content Loss
        content_loss = content_loss_function(generated_layers, chosen_content_layer_nums)

        #Calculate Style Loss
        style_loss = style_loss_function(generated_layers, chosen_style_layer_nums)

        #Total Loss
        total_loss = alpha*content_loss + beta*style_loss

        #Make sure the gradient is zeroed and ready for calculation based on new losses
        optimizer.zero_grad()

        #Back propagate the loss through the image
        total_loss.backward()
        optimizer.step()

        #Generate Animation frame
        if step % 1 == 0:
            print(f"step:{step}, style_loss:{style_loss.item()}, content_loss:{content_loss.item()}")
            
            # input()

            # Add image to animation
            output_image = torch.clamp(generated_image, min=0, max=1)
            # plt.figure()
            display(output_image, ims, ax)
            # plt.savefig(f"/step_{step}_Image")
            # files.download( f"/step_{step}_Image.png" ) 
            # plt.show()

    # Create an animation
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=1000)
    # Save the animation
    ani.save('style_transfer.mp4')

    pass

if __name__=="__main__":
    __ITB__ = AutoFormattedTB(mode = 'Verbose', color_scheme='LightBg', tb_offset = 1)
    main()