import torch
import torch.nn as nn



class AngleClassification(nn.Module):
    def __init__(self, encoder_output_dim, num_angels, metadata_len=0):
        super(AngleClassification, self).__init__()
        self.encoder_output_dim = encoder_output_dim
        self.fc_1 = nn.Linear(encoder_output_dim+metadata_len, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc_2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc_3 = nn.Linear(256, num_angels)
        # self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(p=0.2)
        # self.fc_4 = nn.Linear(256, num_angels)

    def forward(self, x, metadata=None):
        if metadata is not None:
            if metadata.dim() == 1:
                metadata = metadata.unsqueeze(1)  # Convert (batch_size,) to (batch_size, 1)

            # Concatenate along the channel dimension
            x = torch.cat((x, metadata), dim=1)

        z = self.fc_1(torch.flatten(x, start_dim=1))
        z = self.relu1(z)
        z = self.dropout1(z)
        z = self.fc_2(z)
        z = self.relu2(z)
        z = self.dropout2(z)
        z = self.fc_3(z)
        # z = self.relu3(z)
        # z = self.dropout3(z)
        # z = self.fc_4(z)
        return z


class RadiusClassification(nn.Module):
    def __init__(self, encoder_output_dim, num_rads, metadata_len=0):
        super(RadiusClassification, self).__init__()
        self.encoder_output_dim = encoder_output_dim
        self.fc_1 = nn.Linear(encoder_output_dim+metadata_len, num_rads)
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=0.5)
        # self.fc_2 = nn.Linear(64, num_rads)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=0.5)
        # self.fc_3 = nn.Linear(32, num_rads)

    def forward(self, x, metadata=None):
        if metadata is not None:
            if metadata.dim() == 1:
                metadata = metadata.unsqueeze(1)  # Convert (batch_size,) to (batch_size, 1)

            # Concatenate along the channel dimension
            x = torch.cat((x, metadata), dim=1)

        z = self.fc_1(torch.flatten(x, start_dim=1))
        # z = self.relu1(z)
        # z = self.dropout1(z)
        # z = self.fc_2(z)
        # z = self.relu2(z)
        # z = self.dropout2(z)
        # z = self.fc_3(z)
        # z = self.relu3(z)
        # z = self.dropout3(z)
        # z = self.fc_4(z)
        return z



def load_angle_model_from_file(filepath, encoder_output_dim:int, num_locations:int, device:str):
    model = AngleClassification(encoder_output_dim, num_locations).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    return model


