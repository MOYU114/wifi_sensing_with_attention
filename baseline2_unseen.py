# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:13:34 2023

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import trange
import csv
import numpy as np
import pandas as pd

if (torch.cuda.is_available()):
    print("Using GPU for training.")
    device = torch.device("cuda:0")
else:
    print("Using CPU for training.")
    device = torch.device("cpu")

class SpatialEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialEncoderBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1))
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        # self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.avg_pool = nn.AdaptiveAvgPool3d((6, 1, 1))

    def forward(self, x):
        x = self.conv3d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        return x

class SpatialFrequencyModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(SpatialFrequencyModel, self).__init__()

        # Spatial Encoder Blocks with distinct kernel sizes
        self.spatial_encoders = nn.ModuleList([
            SpatialEncoderBlock(in_channels, out_channels, kernel_size)
            for kernel_size in kernel_sizes
        ])

        # 3D convolution layer for comprehensive information
        self.comprehensive_conv = nn.Conv3d(len(kernel_sizes) * out_channels, out_channels, kernel_size=(len(kernel_sizes), 1, 1))

    def forward(self, x):
        # Apply each spatial encoder block
        spatial_features = [encoder(x) for encoder in self.spatial_encoders]
        # for i, feature in enumerate(spatial_features):
        #     print(f"Spatial Feature {i + 1} Shape: {feature.shape}")

        # Concatenate spatial features along the channel dimension
        spatial_features_concat = torch.cat(spatial_features, dim=1)

        # Apply the comprehensive 3D convolution layer
        comprehensive_info = self.comprehensive_conv(spatial_features_concat)
        # print(comprehensive_info.shape)
        # print(x.shape)

        # # Concatenate frequency features with spatial features
        # output = torch.cat([comprehensive_info, x], dim=1)

        return comprehensive_info

class EvolvingAttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_size, dilated_rate):
        super(EvolvingAttentionModule, self).__init__()

        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 1D convolution to generate channel attention mask
        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size, padding=1)

        # GRU layer to evolve attention masks
        self.gru = nn.GRUCell(in_channels, in_channels)

        # Parameters for dilated rate
        self.dilated_rate = dilated_rate

    def forward(self, x):
        # Global Average Pooling (GAP) along the sequence dimension
        global_info = self.global_avg_pool(x).squeeze(dim=1)
        global_info = global_info.squeeze(dim=-1)
        global_info = global_info.squeeze(dim=-1)

        # 1D convolution to generate channel attention mask
        channel_attention_mask = self.conv1d(global_info)
        channel_attention_mask = channel_attention_mask.squeeze(-1)

        # GRU layer to evolve attention masks
        evolving_attention = []
        h_t = torch.zeros_like(channel_attention_mask)
        
        for t in range(x.size(1)):
            # Apply GRU to evolve attention masks
            h_t = self.gru(channel_attention_mask, h_t)
            evolving_attention.append(h_t)

        evolving_attention = torch.stack(evolving_attention, dim=1)

        return evolving_attention
    
class FeatureDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureDecoder, self).__init__()

        # 3D convolution layer to map enhanced features to output feature maps
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.linear = nn.Linear(128, 28)

    def forward(self, x):
        x = x.view(len(x),64,64,1,1)
        # print(x.shape)
        # 3D convolution to generate output feature maps
        output_feature_maps = self.conv3d(x)
        temp = output_feature_maps.view(len(output_feature_maps),-1)
        output_feature_maps = self.linear(temp)
        # print(output_feature_maps.shape)

        return output_feature_maps

class PoseLoss(nn.Module):
    def __init__(self, lambda_jhm=1.0, lambda_paf=1.0):
        super(PoseLoss, self).__init__()
        self.lambda_jhm = lambda_jhm
        self.lambda_paf = lambda_paf
        self.criterion = nn.MSELoss()

    def forward(self, predicted_jhm, target_jhm):#, predicted_paf, target_paf):
        # Joint Heatmaps (JHMs) loss
        loss_jhm = self.criterion(predicted_jhm, target_jhm)

        # Part Affinity Fields (PAFs) loss
        # loss_paf = self.criterion(predicted_paf, target_paf)

        # Total loss
        total_loss = self.lambda_jhm * loss_jhm# + self.lambda_paf * loss_paf

        return total_loss
    
def reshape_and_average(x):
    num_rows = x.shape[0]
    averaged_data = np.zeros((num_rows, 50))
    for i in trange(num_rows):
        row_data = x.iloc[i].to_numpy()
        reshaped_data = row_data.reshape(-1, 50)
        reshaped_data = pd.DataFrame(reshaped_data).replace({None: np.nan}).values
        reshaped_data = pd.DataFrame(reshaped_data).dropna().values
        non_empty_rows = np.any(reshaped_data != '', axis=1)
        filtered_arr = reshaped_data[non_empty_rows]
        reshaped_data = np.asarray(filtered_arr, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    return averaged_df

def fillna_with_previous_values(s):
    non_nan_values = s[s.notna()].values
    nan_indices = s.index[s.isna()]
    n_fill = len(nan_indices)
    n_repeat = int(np.ceil(n_fill / len(non_nan_values)))
    fill_values = np.tile(non_nan_values, n_repeat)[:n_fill]
    s.iloc[nan_indices] = fill_values
    return s
'''
CSI_test = "./data/static/data/device/CSI_static_6C.csv"
Video_test = "./data/static/data/device/points_static.csv"
with open(CSI_test, "r") as csvfilee:
    csvreadere = csv.reader(csvfilee)
    data2 = list(csvreadere)  # 将读取的数据转换为列表
csi_test = pd.DataFrame(data2)
test_bb = reshape_and_average(csi_test)
test_bb = test_bb.values.astype('float32')
csi_test = test_bb / np.max(test_bb)
video_test = pd.read_csv(Video_test, header=None)
video_test = video_test.values.astype('float32')
video_test = video_test.reshape(len(video_test), 14, 2)
video_test = video_test / [1280, 720]
video_test = video_test.reshape(len(video_test), -1)
data = np.hstack((Video_test, CSI_test))

# b = torch.from_numpy(csi_test).double()
# b = b.view(len(b),int(len(csi_test[0])/10),10)
# g = torch.from_numpy(video_test).double()

original_length = video_test.shape[0]

# 创建模型实例
# in_channels = 64  # 输入通道数，应与前一个模块的输出通道数匹配
# kernel_size = 3  # 1D 卷积的核大小
# dilated_rate = 2  # GRU dilated rate
evolving_attention_module = EvolvingAttentionModule(64, 3, 2)


# 创建模型实例
in_channels = 1  # 输入通道数
out_channels = 64  # 输出通道数
kernel_sizes = [3, 5]  # 不同分支的卷积核大小
spatial_frequency_model = SpatialFrequencyModel(in_channels, out_channels, kernel_sizes).to(device)

# Create instances of the model, feature decoder, and loss
easfn_model = SpatialFrequencyModel(in_channels=1, out_channels=64, kernel_sizes=[3,5]).to(device) #[5, 10, 15, 20, 25]
evolving_attention_module = EvolvingAttentionModule(in_channels=64, kernel_size=3, dilated_rate=2).to(device)
feature_decoder = FeatureDecoder(in_channels=64, out_channels=2).to(device)
pose_loss = PoseLoss(lambda_jhm=1.0, lambda_paf=1.0).to(device)

# Set up the optimizer and learning rate scheduler
optimizer = optim.Adam(easfn_model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
num_epochs = 200
batch_size = 4

for epoch in range(num_epochs):
    easfn_model.train()
    
    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    f = torch.from_numpy(video_test[random_indices, :]).to(device).float()
    a = torch.from_numpy(csi_test[random_indices, :]).to(device).float()
    a = np.hstack([a,a])
    f = f.view(batch_size, 2, 14)
    a = a.reshape(batch_size, 1, 5, 5, 4)
    a = torch.from_numpy(a)
    
    spatial_frequency_output = easfn_model(a)
    evolving_attention_output = evolving_attention_module(spatial_frequency_output)
    feature_decoder_output = feature_decoder(evolving_attention_output)
    feature_decoder_output = feature_decoder_output.view(batch_size,2,14)

    # Calculate loss
    loss = pose_loss(feature_decoder_output, f).float()  # Assuming targets are your ground truth JHMs and PAFs

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # for batch_idx, inputs in enumerate(train_dataloader):
    #     # Forward pass
    #     spatial_frequency_output = easfn_model(inputs)
    #     evolving_attention_output = evolving_attention_module(spatial_frequency_output)
    #     feature_decoder_output = feature_decoder(evolving_attention_output)

    #     # Calculate loss
    #     loss = pose_loss(feature_decoder_output, targets)  # Assuming targets are your ground truth JHMs and PAFs

    #     # Backward pass and optimization
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # # Learning rate scheduling step
    # scheduler.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# torch.save(evolving_attention_module.state_dict(), './model/baseline1_evo.pth')
torch.save(spatial_frequency_model.state_dict(), './model/baseline1_spa1.pth')
torch.save(easfn_model.state_dict(), './model/baseline1_easfn1.pth')
torch.save(evolving_attention_module.state_dict(), './model/baseline1_evolving1.pth')
torch.save(feature_decoder.state_dict(), './model/baseline1_feature1.pth')
print("Training complete.")
'''
# # Assuming you have your test dataset and dataloader
# # test_dataset = YourTestDataset(...)
# # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Set the model to evaluation mode
# easfn_model.eval()

# # Initialize variables to store predictions and targets for analysis
# all_predictions = []
# all_targets = []

# # Testing loop
# with torch.no_grad():
#     for batch_idx, test_inputs in enumerate(test_dataloader):
#         # Forward pass
#         spatial_frequency_output = easfn_model(test_inputs)
#         evolving_attention_output = evolving_attention_module(spatial_frequency_output)
#         feature_decoder_output = feature_decoder(evolving_attention_output)

#         # Store predictions and targets for analysis
#         all_predictions.append(feature_decoder_output.cpu().numpy())  # Assuming predictions are NumPy arrays
#         all_targets.append(test_targets.cpu().numpy())  # Assuming targets are NumPy arrays

# # Concatenate predictions and targets along the batch dimension
# all_predictions = np.concatenate(all_predictions, axis=0)
# all_targets = np.concatenate(all_targets, axis=0)

# # Perform analysis on predictions and targets as needed
# # ...

# # For example, you can calculate performance metrics, visualize results, etc.
# # You might use tools like scikit-learn, matplotlib, or other relevant libraries for analysis.
# '''
in_channels = 1  # 输入通道数
out_channels = 64  # 输出通道数
kernel_sizes = [3, 5]  # 不同分支的卷积核大小
spatial_frequency_model = SpatialFrequencyModel(in_channels, out_channels, kernel_sizes).to(device)
easfn_model = SpatialFrequencyModel(in_channels=1, out_channels=64, kernel_sizes=[3,5]).to(device) #[5, 10, 15, 20, 25]
evolving_attention_module = EvolvingAttentionModule(in_channels=64, kernel_size=3, dilated_rate=2).to(device)
feature_decoder = FeatureDecoder(in_channels=64, out_channels=2).to(device)

spatial_frequency_model.load_state_dict(torch.load('./model/baseline1_spa.pth'))
easfn_model.load_state_dict(torch.load('./model/baseline1_easfn.pth'))
evolving_attention_module.load_state_dict(torch.load('./model/baseline1_evolving.pth'))
feature_decoder.load_state_dict(torch.load('./model/baseline1_feature.pth'))

for i in range(1):
    if i == 0:
        Video_test = "./data/static/data/device/points_arm_left.csv"
        CSI_test = "./data/static/data/device/CSI_arm_left_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 1:
        Video_test = "./data/static/data/device/points_wave_right_6C.csv"
        CSI_test = "./data/static/data/device/CSI_wave_right_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==2:
        Video_test = "./data/static/data/device/points_leg_left_6C.csv"
        CSI_test = "./data/static/data/device/CSI_leg_left_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    else:
        Video_test = "./data/static/data/device/points_leg_right_6C.csv"
        CSI_test = "./data/static/data/device/CSI_leg_right_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    with open(CSI_test, "r") as csvfilee:
        csvreadere = csv.reader(csvfilee)
        data2 = list(csvreadere)  # 将读取的数据转换为列表
    csi_test = pd.DataFrame(data2)
    test_bb = reshape_and_average(csi_test)
    test_bb = test_bb.values.astype('float32')
    csi_test = test_bb / np.max(test_bb)
    video_test = pd.read_csv(Video_test, header=None)
    video_test = video_test.values.astype('float32')
    video_test = video_test.reshape(len(video_test), 14, 2)
    video_test = video_test / [1280, 720]
    video_test = video_test.reshape(len(video_test), -1)
    data = np.hstack((Video_test, CSI_test))
    original_length = video_test.shape[0]
    
    
    with torch.no_grad():   
        g = torch.from_numpy(video_test).to(device).float()
        a = torch.from_numpy(csi_test).to(device).float()
        a = np.hstack([a,a])
        # f = f.view(batch_size, -1)
        a = a.reshape(len(a), 1, 5, 5, 4)
        a = torch.from_numpy(a)
        
        spatial_frequency_output = easfn_model(a)
        evolving_attention_output = evolving_attention_module(spatial_frequency_output)
        feature_decoder_output = feature_decoder(evolving_attention_output)
        r = feature_decoder_output.view(len(feature_decoder_output),-1)
        gnp = g.numpy()
        rnp = r.numpy()
        np.savetxt(Video_OUTPUT_PATH, gnp, delimiter=',')
        np.savetxt(CSI_OUTPUT_PATH, rnp, delimiter=',')
# '''