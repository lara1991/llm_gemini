import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from icecream import ic


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_midas_model(model_type = "MiDaS_small"):
    model = torch.hub.load("intel-isl/MiDaS",model_type)

    model.to(DEVICE)
    model.eval()

    return model

def load_transformations():
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return transform

def load_and_process_image(img,transform):
 
    img = np.asarray(img)
    # ic(img.shape)

    input_batch = transform(img).to(DEVICE)
    return input_batch

def model_inferencing_and_post_processing(inputs,model,img_size):
    with torch.no_grad():
        prediction = model(inputs)
    
    # print(inputs.shape)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        img_size,
        mode='bicubic',
        align_corners=False,
    ).squeeze()

    output = prediction.cpu().numpy()
    return output


def get_depth_map_image(image=None,midas_model=None,midas_tranforms=None,):
    
    # midas_model = load_midas_model(model_type="DPT_Hybrid")
    # midas_transforms = load_transformations()

    # img_file_path = "images/london-street-view-songquan-deng.jpg"
    # pil_image = Image.open(image)
    # pil_image = pil_image.convert("RGB")

    img_size = image.size
    # print(img_size)
    model_inputs = load_and_process_image(
        img=image,
        transform=midas_tranforms,
    )
    
    output = model_inferencing_and_post_processing(
        inputs = model_inputs,
        model=midas_model,
        img_size = (img_size[1],img_size[0]),
    )
    output = output.astype(np.uint8)
    # print(output)
    # print(output.shape)
    pil_depth_image = Image.fromarray(output)
    # print(type(pil_image))
    # plt.imshow(output)
    # # ic(output)
    # plt.show()

    return pil_depth_image


if __name__=="__main__":
    # get_depth_map_image()
    pass
