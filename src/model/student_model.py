class StudentAttention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)
        return (input * alpha).sum(dim=-2)


class StudentCRNN(nn.Module):

    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config

        self.quant = torch.quantization.QuantStub()

        self.conv = nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            )
        self.relu= nn.ReLU()
        
        self.flatten=nn.Flatten(start_dim=1, end_dim=2)

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
            config.stride[0] + 1
        
        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=0.1,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, input):
        input=self.quant(input)
        input = input.unsqueeze(dim=1)
        conv_output = self.flatten(self.conv(input)).transpose(-1, -2)

        conv_output = self.dequant(conv_output)
        gru_output, _ = self.gru(conv_output)
        gru_output = self.quant(gru_output)

        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        output = self.dequant(output)
        return output