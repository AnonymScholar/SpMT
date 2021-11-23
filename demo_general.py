import os
import time
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.networks.face_parsing.parsing_model import BiSeNet

opt = DemoOptions().parse()

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

def denorm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


model = Pix2PixModel(opt)

if len(opt.gpu_ids) > 0:
            model = DataParallelWithCallback(model,device_ids=opt.gpu_ids)
model.eval()


n_classes = 19
parsing_net = BiSeNet(n_classes=n_classes)
parsing_net.load_state_dict(torch.load('./models/networks/face_parsing/79999_iter.pth'))
parsing_net.eval()
for param in parsing_net.parameters():
    param.requires_grad = False

if not os.path.exists(opt.result_dir):
    os.mkdir(opt.result_dir)


makeup_names = [name.strip() for name in open(os.path.join(opt.dataroot,'makeup_test.txt'), "rt").readlines()]
non_makeup_names=[name.strip() for name in open(os.path.join(opt.dataroot,'non-makeup_test.txt'), "rt").readlines()]

for i in range(opt.demo_nums):
    print(i,'/',opt.demo_nums,' demo')
    makeup_name=makeup_names[i] 
    non_makeup_name=non_makeup_names[i]

    c = Image.open(os.path.join(opt.dataroot,'images/non-makeup',non_makeup_name)).convert('RGB')
    s = Image.open(os.path.join(opt.dataroot,'images/makeup',makeup_name)).convert('RGB')

    opt.height, opt.width = 256, 256*c.size[0]//c.size[1]
    s_height, s_width = 256, 256*s.size[0]//s.size[1]
    c_m = c.resize((512, 512))
    s_m = s.resize((512,512))
    c = c.resize((256,256))
    s = s.resize((256,256))
    
    c_tensor = trans(c).unsqueeze(0)
    s_tensor = trans(s).unsqueeze(0)
    c_m_tensor = trans(c_m).unsqueeze(0)
    s_m_tensor = trans(s_m).unsqueeze(0)

    x_label = parsing_net(c_m_tensor)[0]  
    y_label = parsing_net(s_m_tensor)[0]
    x_label=F.interpolate(x_label, (opt.height, opt.width), mode='bilinear', align_corners=True)
    y_label=F.interpolate(y_label, (opt.height, opt.width), mode='bilinear', align_corners=True)
    x_label = torch.softmax(x_label, 1) 
    y_label = torch.softmax(y_label, 1)

    nonmakeup_unchanged = (x_label[0,0,:,:]+x_label[0,4,:,:]+x_label[0,5,:,:]+x_label[0,11,:,:]+x_label[0,16,:,:]+x_label[0,17,:,:]).unsqueeze(0).unsqueeze(0)
    makeup_unchanged = (y_label[0,0,:,:]+y_label[0,4,:,:]+y_label[0,5,:,:]+y_label[0,11,:,:]+y_label[0,16,:,:]+y_label[0,17,:,:]).unsqueeze(0).unsqueeze(0)

    input_dict = {'nonmakeup': c_tensor,
              'makeup': s_tensor,
              'label_A': x_label,
              'label_B': y_label,
              'makeup_unchanged': makeup_unchanged,
              'nonmakeup_unchanged': nonmakeup_unchanged
              }


    time_start = time.time()

    synthetic_image = model([input_dict], mode='inference')

    time_end = time.time()
    print(time_end - time_start)

    out = denorm(synthetic_image[0])
    out = F.interpolate(out, (opt.height, opt.width), mode='bilinear', align_corners=False)
    out_demo = torch.cat([denorm(c_tensor),denorm(s_tensor),out[0].unsqueeze(0).cpu()],3)

    c_name = os.path.splitext(os.path.basename(non_makeup_name))[0]
    s_name = os.path.splitext(os.path.basename(makeup_name))[0]
    opt.output_name = f'{opt.result_dir}/{c_name}_{s_name}'

    save_image(out, f'{opt.output_name}.jpg', nrow=1)
    save_image(out_demo,f'{opt.output_name}_style_transfer_demo.jpg')

    print(f'result saved into files starting with {opt.output_name}\n')