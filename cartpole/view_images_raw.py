import data
import models
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

imgs = data.load_np("img_dat.gz", "data_full")

batch = 0

raw_img = torch.tensor(imgs[batch*256:(batch+1)*256], dtype=torch.float32).swapaxes(1, 3).swapaxes(2, 3) / 256

out_imgs = []

for i in range(256):
    print(imgs[i].shape)
    pilimg = transforms.ToPILImage()(raw_img[i])
    print(torch.mean(raw_img[i]).item())
    #print(torch.nn.MSELoss()(torch.tensor(enc)[i], torch.tensor(enc)[i+1]))
    dst = pilimg
    #dst = transforms.ToPILImage()(torch.tensor(raw_img[i]))
    out_imgs.append(dst)

out_imgs[0].save("out_test1.gif", save_all=True, append_images=out_imgs[1:], duration=10, loop=0)
