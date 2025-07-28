from PIL import Image, ImageTk
import torch
import models
import torchvision.transforms as transforms
import tkinter as tk
from config import lat

model = models.load_ae("model_ae_8_interp.torch")    
model_lstm = models.load_lstm("model_lstm_8_interp.torch")

model.eval()
model_lstm.eval()

model_lstm.set_batch_size(1)
model_lstm.zero_hidden()

window = tk.Tk()
window.title("Cursed QWOP")

label = tk.Label(window)
label.pack()

key_inputs = [0]
pressed = False

reset_state = torch.tensor([[1.6396663188934326, -1.799014925956726, 0.2766458988189697, 2.800827741622925, 2.989149808883667, -1.7274534702301025, -1.4265387058258057, -0.7298376560211182, 0]])

mu_diff = torch.tensor([ 2.4264e-03, -2.7636e-03,  4.4383e-05,  2.2945e-03,  3.8747e-03,
        -9.0996e-04, -1.4763e-03,  2.0510e-03])

sigma_diff = torch.tensor([0.6051, 0.5535, 0.5855, 0.6326, 0.9133, 0.6024, 0.4599, 0.7968])

def onKeyPress(event):
    global key_inputs
    global state
    global model_lstm
    global pressed

    key = event.char
    if key == 'r':
        state = reset_state
        model_lstm.zero_hidden()
    if key == 'j':
        key_inputs[0] = 0
        pressed = True
    if key == 'k':
        key_inputs[0] = 1
        pressed = True

def onKeyRelease(event):
    global key_inputs
    global pressed

    key = event.keycode

    pressed = False

state = reset_state
def update_img(i):
    global state
    global key_inputs
    model_lstm.eval()
    model.eval()
    with torch.no_grad():
        output_img = transforms.ToPILImage()(model.decode(state[[0],:lat])[0]).resize((256,256), Image.Resampling.BICUBIC)
        output_tk_img = ImageTk.PhotoImage(output_img)
        state = torch.unsqueeze(state, 0)
        lstm_output = (mu_diff + model_lstm(state) * sigma_diff)[0] + state[0,:,:lat]

        label.pil_image = output_img
        label.image = output_tk_img
        label.config(image=output_tk_img)
        label.pack()

        new_state = torch.cat((lstm_output, torch.tensor([key_inputs], dtype=torch.float32)), dim=-1)
        state = new_state

        print(key_inputs)
        if not pressed:
            key_inputs[0] = 1-key_inputs[0]

        if i < 10000:
            window.after(int(20), update_img, i+1)

update_img(0)
window.bind("<KeyPress>", onKeyPress)
window.bind("<KeyRelease>", onKeyRelease)
window.mainloop()