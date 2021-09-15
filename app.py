from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
import gradio as gr

def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb


import cv2

def segment(img, show_orig=True, dev='cuda'):
  net = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

  to_pil_image = torchvision.transforms.ToPILImage()
  img = to_pil_image(img)
  #img = Image.open(path)
  #if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = T.Compose([T.Resize(640), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  rgb = decode_segmap(om)
  return rgb
  #plt.imshow(rgb); plt.axis('off'); #plt.show()

#fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
#dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

#img = Image.open("img1.jpg")
#img = cv2.imread("img1.jpg")
#segment(img)

gr_interface = gr.Interface(fn=segment, inputs=gr.inputs.Image(shape=(1080, 1080)), outputs="image")
gr_interface.launch(debug=True)

