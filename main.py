import toml
import os
import torch
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
config = toml.load("./config.toml")

#用于对不适当的扫描仪照明进行归一化处理. 参考readme
INCIDENT_LIGHT = torch.FloatTensor(config["data"]["incident_light"]).to(torch.device(config["device"]))

from Models import FeatureNet, TDSS, ReconstructionLayer
from utils import absorption_criterion, plot_loss, RGB2OpticalDensity, OpticalDensity2RGB
from Data import PatchDataset
from torch.utils.data import DataLoader
from torchvision import utils as vutils


"""
[0] Datasets, dataloaders, data transforms的定义
"""

#是否要自定义transform
if bool(config["training_data"]["use_custom_transform"]):
    from Data import CustomTransform as TrainingTransform
else:
    from Data import DefaultTransform as TrainingTransform

# 构建训练集

training_dataset = PatchDataset(config["training_data"]["path_to_patches"], TrainingTransform())

#构建dataloader
training_dataloader = DataLoader(
    training_dataset,
    batch_size=int(config["training_data"]["batch_size"]),
    shuffle=True,
    # num_workers=int(config["training_data"]["num_workers"]),
    drop_last=True
)




"""
[1] 以下为特征提取过程
    - 如果force_train参数为true, 那么特征提取模型肯定要进行训练
    - 如果预训练force_train为false 并且path_to_state 已经被给出, 那么模型将会被加载，训练过程直接跳过.
    - 如果预训练force_train为false 并且path_to_state 未被给出, 就会报错
"""
feature_config = config["feature_extraction"]
feature_net = FeatureNet(**feature_config).to(torch.device(config["device"]))
feature_net_state_path = config["feature_extraction"]["path_to_state"]

if bool(feature_config["force_training"]):
    # 进行训练

    recon_layer = ReconstructionLayer(
        int(config["feature_extraction"]["multiplier"]),
        int(config["feature_extraction"]["input_channels"])
    ).to(config["device"])

    criterion = nn.MSELoss()

    learning_rate = float(config["feature_extraction"]["learning_rate"])

    params = chain(feature_net.parameters(), recon_layer.parameters())
    optimiser = optim.Adam(params, lr=learning_rate)

    num_epochs = int(config["feature_extraction"]["training_epochs"])
    print("pre-training feature extractor")
    for _ in tqdm(range(num_epochs)):
        for data in training_dataloader:

            data = data.to(torch.device(config["device"]))

            # 通过incident light进行归一化操作
            data = (data / INCIDENT_LIGHT.view(1, 3, 1, 1)).clamp(max=1.0)

            data_od = RGB2OpticalDensity(data)

            optimiser.zero_grad()

            _, features = feature_net(data_od)
            reconstruction = recon_layer(features)

            loss = criterion(reconstruction, data_od)
            loss.backward()

            optimiser.step()

    torch.save(feature_net.state_dict(), feature_net_state_path)

else:
    # 这里加了一个断言来检测是否存在已经训练好的模型
    assert len(feature_net_state_path) > 0 and os.path.exists(feature_net_state_path), "feature_extraction.path_to_state is not valid."

    feature_net.load_state_dict(torch.load(feature_net_state_path))


"""
[2] 染色分离过程
"""
#======================================进行网络模型的搭建=============================================
tdss = TDSS(
    feature_net,
    int(config["feature_extraction"]["multiplier"]),
    int(config["stain_separation"]["number_of_stains"]),
    float(config["stain_separation"]["alpha"])
).to(torch.device(config["device"]))

reconstruction_criterion = nn.L1Loss()
optimiser_tdss = optim.Adam(tdss.parameters(), lr=float(config["stain_separation"]["learning_rate"]))

loss_log = {}
loss_log["reconstruction"] = []
loss_log["sparsity"] = []

print("training TDSS")
for epoch in tqdm(range(int(config["stain_separation"]["training_epochs"]))):
    for batch_id, data in enumerate(training_dataloader):

        optimiser_tdss.zero_grad()

        data = data.to(torch.device(config["device"]))



        # 通过incident light进行归一化操作
        data = data / INCIDENT_LIGHT.view(1, 3, 1, 1)        

        data_od = RGB2OpticalDensity(data)


        reconstructed_stains, _ = tdss(data_od)


        # 将各自的染色进行合并来重建之前染色图片x
        reconstruction = torch.stack(reconstructed_stains, dim=0).sum(dim=0)



        reconstruction_loss = reconstruction_criterion(OpticalDensity2RGB(reconstruction), data)
        sparsity_loss = absorption_criterion(reconstructed_stains)

        #print("reconstruction loss", reconstruction_loss.item(), "sparsity_loss", sparsity_loss.item())
        lambda1 = float(config["stain_separation"]["sparsity_weight"])
        #定义loss为重建loss + lambda * 稀疏损失
        loss = reconstruction_loss + (lambda1 * sparsity_loss)
        loss.backward()

        optimiser_tdss.step()

        loss_log["reconstruction"].append([epoch, batch_id, reconstruction_loss.item()])
        loss_log["sparsity"].append([epoch, batch_id, sparsity_loss.item()])

        batch_id += 1

    # 画出损失值
    
    plot_loss(loss_log["reconstruction"], "./output/training/reconstruction_loss.png")
    plot_loss(loss_log["sparsity"], "./output/training/absorption_loss.png")

    if epoch % 5 == 0:
        # 每过5轮就保存一次batch_size个图片的拆解结果
        for s, stain_od in enumerate(reconstructed_stains):
            stain_rgb = OpticalDensity2RGB(stain_od)

            vutils.save_image(
                stain_rgb.detach(),
                "./output/training/reconstructed_stain_{}.png".format(s),
                normalize=False
            )


        vutils.save_image(
            data.detach(),
            "./output/training/data.png",
            normalize=False
        )

        reconstruction_rgb = OpticalDensity2RGB(reconstruction)
        vutils.save_image(
            reconstruction_rgb.detach(),
            "./output/training/reconstruction.png",
            normalize=False
        )


    # 保存模型参数至condig.toml中的路径
    torch.save(tdss.state_dict(), config["stain_separation"]["path_to_state"])