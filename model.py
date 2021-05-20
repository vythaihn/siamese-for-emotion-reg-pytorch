import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):

    #TODO: Add Batch Norm
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 94),  # 96x35x35
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride = 2),  # 96x17x17
            nn.Conv2d(96, 256, 7, stride = 1, padding= 3),
            nn.ReLU(),    # 256x17x17
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride = 2),   # 256x8x8
            nn.Conv2d(256, 384, 5, padding = 2, stride = 1), #384x8x8
            nn.ReLU(), # 128@18*18
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, 5, padding=2, stride=1), #256x8x8
            nn.ReLU(),  # 128@18*18
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride = 2) # 256x4x4
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        self.task_A_concat = nn.Sequential(
            nn.Linear(12288, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 3)
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def taskA_one(self, x):
        x = self.task_A_concat(x)
        return x

    def forward(self, x):
        x1,x2,x3 = x

        #print(x1.size())
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out3 = self.forward_one(x3)

        concat1 = torch.cat((torch.tensor(out1), torch.tensor(out2), torch.tensor(out3)), 1)
        concat2 = torch.cat((torch.tensor(out2), torch.tensor(out3), torch.tensor(out1)), 1)
        concat3 = torch.cat((torch.tensor(out3), torch.tensor(out1), torch.tensor(out2)), 1)

        x1 = self.taskA_one(concat1)
        x2 = self.taskA_one(concat2)
        x3 = self.taskA_one(concat3)

        task_A_out = (x1,x2,x3)
        return task_A_out


# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(net):,} trainable parameters')
