import torch.nn as nn


class Layer(nn.Module):

    def __init__(self,in_channels,out_channels):

        super(Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2)
        self.bnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2,2)


    def forward(self,x):

        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.mp(x)

        return x
    
    
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
          
        self.conv_layer_1=Layer(in_channels=3, out_channels=12)
        self.conv_layer_2=Layer(in_channels=12,  out_channels=24)
        self.conv_layer_3=Layer(in_channels=24,  out_channels=36)
        self.conv_layer_4=Layer(in_channels=36,  out_channels=48)
        self.conv_layer_5=Layer(in_channels=48,  out_channels=36)
        self.conv_layer_6=Layer(in_channels=36,  out_channels=24)
        self.conv_layer_7=Layer(in_channels=24,  out_channels=12)
        self.conv_layer_8=Layer(in_channels=12,  out_channels=4)
        
        self.net = nn.Sequential(self.conv_layer_1,self.conv_layer_2,self.conv_layer_3, self.conv_layer_4, 
                                 self.conv_layer_5, self.conv_layer_6, self.conv_layer_7, self.conv_layer_8)
             
    def forward(self, x):
        
        in_size=x.size(0)
        x = self.net(x.float())
        x = x.view(in_size, -1)
        return x


class CnnLstm(nn.Module):
    def __init__(self):
        super(CnnLstm, self).__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=160,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(160, 1)

    def forward(self, x):
        batch_size, channels, time_steps, height, width = x.size()
        c_in = x.view(batch_size * time_steps, channels, height, width)
        c_out = self.cnn(c_in)

        r_in = c_out.view(batch_size, time_steps, -1)
        r_out, (_, _) = self.lstm(r_in)
        r_out2 = self.linear(r_out[:, -1, :])
        return r_out2
