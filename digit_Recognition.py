from tkinter import *
from tkinter import ttk
from ttkbootstrap import *
from tkinter import filedialog as fd
from PIL import ImageTk,Image
import torch.nn as nn 
import torch
import  torchvision.transforms as transforms

class Netto(nn.Module):
    def __init__(self):
        super(Netto, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # Image size 32 x32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # Image size 16 x16
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # Image size 8 x8 = 64
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x2 = self.relu4(x)
        x = self.fc3(x2) 
        return x
device = torch.device('cpu')
model = Netto()
model.load_state_dict(torch.load('Newmnist.pth',map_location=device))

class window:
    def __init__(self,root,title) -> None:
        self.root=root 
        self.title=title
        self.root.resizable(0,0)
        self.root.geometry('700x500')
        self.canvas = Canvas(self.root,width=400,height=400)
        self.canvas.pack()
        self.run = ttk.Button(self.root,text = 'Start',command=self.start)
        self.run.pack(side='right',padx=10)
        self.openfile = ttk.Button(self.root,text='Browse Image',command=self.select_file)
        self.openfile.pack(side='right',padx=10)
        self.pred_label = Label(self.root,text='',font=('Arial',25))

        
    
    def select_file(self):
        try:
            self.filepath = fd.askopenfilename()
            # print(filename)
            self.img0 = Image.open(self.filepath)
            self.img = ImageTk.PhotoImage(self.img0.resize((400,400),Image.ANTIALIAS))
            self.canvas.create_image(20,20,anchor=NW,image=self.img)
            self.canvas.image=self.img
        except:
            pass
    def start(self):
        self.pred_label.config(text='')
        self.pred_label.pack()
        model.eval()
        transform = transforms.Compose([transforms.PILToTensor()])
        img_tensor = transform(self.img0)
        image = img_tensor.unsqueeze(0)
        # print(image.shape,image.device,image.dtype)
        image=image.float()
        
        # print(next(model.parameters()).is_cuda) # this is to check if 
        # the weights are on cpu or gpu    
        output = model(image)
        # print(output)
        pred_val,pred_i = output.max(1)
        self.pred_label.config(text=f'Label predicted: {pred_i.item()}')
        

if __name__ == '__main__':
    win = Style(theme='darkly').master
    obj = window(win,'Digit Recognizer')
    win.mainloop()