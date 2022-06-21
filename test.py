import toml
import os
import torch

config = toml.load("./config.toml")

from Models import FeatureNet, TDSS, ReconstructionLayer
from utils import absorption_criterion, plot_loss, RGB2OpticalDensity, OpticalDensity2RGB, mkdir, save_image
from Data import PatchDataset
from torch.utils.data import DataLoader



"""
[0] Datasets, dataloaders, data transforms
"""

# check if a custom transform is needed
if bool(config["training_data"]["use_custom_transform"]):
    from Data import CustomTransform as TrainingTransform
else:
    from Data import DefaultTransform as TrainingTransform


INCIDENT_LIGHT = torch.FloatTensor(config["data"]["incident_light"]).to(torch.device(config["device"]))

# build the testing dataset
testing_dataset = PatchDataset(config["testing_data"]["path_to_patches"], TrainingTransform())
#collate_fn对应的函数,将元组类型的bbb传入得到batch,一共32个数据元组，对应着tensor图片数据以及target，target也即文件名
def collate_fn(bbb):
    a1 = []
    a2 = []
    for v1, v2 in bbb:
        a1.append(v1)
        a2.append(v2)
    a2 = torch.stack(a2)
    return a1, a2
# build the testing data loader
testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=int(config["testing_data"]["batch_size"]),
    shuffle=True,
    num_workers=int(config["testing_data"]["num_workers"]),
    collate_fn=collate_fn,
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

assert len(feature_net_state_path) > 0 and os.path.exists(feature_net_state_path), "feature_extraction.path_to_state is not valid."

feature_net.load_state_dict(torch.load(feature_net_state_path))

"""
[2] 染色分离过程
"""
tdss = TDSS(
    feature_net,
    int(config["feature_extraction"]["multiplier"]),
    int(config["stain_separation"]["number_of_stains"]),
    float(config["stain_separation"]["alpha"])
).to(torch.device(config["device"])).eval()
tdss_net_state_path = config["stain_separation"]["path_to_state"]


assert len(tdss_net_state_path) > 0 and os.path.exists(tdss_net_state_path), "stain_separation.path_to_state is not valid"

tdss.load_state_dict(torch.load(tdss_net_state_path))

output_root = "./output/testing/"
count = 0
images_list=os.listdir(config["testing_data"]["path_to_patches"])

for batch_id, (names,data) in enumerate(testing_dataloader):

    data = data.to(torch.device(config["device"]))

    # 通过incident light进行归一化操作
    data = (data / INCIDENT_LIGHT.view(1, 3, 1, 1)).clamp(max=1.0)


    # 将data转换为光学密度图
    data_od = RGB2OpticalDensity(data)

    # 获取染色图，灰度图
    with torch.no_grad():
        reconstructed_stains, densities = tdss(data_od)



    # 實現對於x的重建
    reconstruction = torch.stack(reconstructed_stains, dim=0).sum(dim=0)

    reconstruction_rgb = OpticalDensity2RGB(reconstruction)

    #print("data", data.size())

    # 获取所有元素，这里不需要的重建图以及染色图已经被注释掉了，只保留了单通道灰度图
    for i in range(data.size(0)):

        item_root_dir = mkdir(os.path.join(output_root, str(count)))
        item_root_dir2 = "/media/totem_disk/totem/ChengYu/TDSS拆解结果/Masson拆解结果/masson测试集拆解结果_红蓝色均衡训练集/"
        # 原图
        data_path = os.path.join(item_root_dir2, "{}data.png".format(str(count)))
        save_image(data[i, ...], data_path)


        # 重建图
        # reconstruction_rgb_path = os.path.join(item_root_dir, "reconstruction.png")
        # save_image(reconstruction_rgb[i, ...], reconstruction_rgb_path)

        # 染色图
        # for s, stain in enumerate(reconstructed_stains):
        #     stain_path = os.path.join(item_root_dir2, "{}stain{}.png".format(count, s))
        #     stain_rgb = OpticalDensity2RGB(stain)
        #
        #     save_image(stain_rgb[i, ...], stain_path)

        #拆解之后的灰度图,通过testing_dataloader中获取到的names来保存路径
        for j in range(densities.size(1)):
            density = densities[i, j, ...]

            density_path = os.path.join(item_root_dir2, "{}.png".format(names[i]))
            save_image(density, density_path)

        count += 1