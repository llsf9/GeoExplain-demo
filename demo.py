"""
1206: 添加自动识别GPS功能
1211: 改变loss的计算方式
1212: 添加编辑模式
"""

import streamlit as st
from PIL import Image
import time
import numpy as np
from utils import * 
from ImgProcess import * 
from model import * 
from visualize import *
from torchvision import transforms


# Palette
from core.util import tensor2img
from paletteModel.network import Network
import json

@st.cache_resource
def load_model(checkpoint, hparams) :
    model = loadModel(checkpoint, hparams)
    return model

@st.cache_resource
def load_edit_model(args, model_pth) :
    paletteModel = Network(**args)
    state_dict = torch.load(model_pth)
    paletteModel.load_state_dict(state_dict, strict=False)

    device = torch.device('cuda:0')
    paletteModel.to(device)
    paletteModel.set_new_noise_schedule(phase='test')
    paletteModel.eval()

    return paletteModel

@st.cache_data
def load_img(uploaded_file) :
    image = Image.open(uploaded_file)
    return image  

def load_base_img(image):
    tfs = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # read input image
    image = image.convert('RGB')
    image = tfs(image)

    return image


def inpainting(img, mask, model):

    mask = torch.from_numpy(mask).permute(2, 0, 1)
    cond_image = img*(1. - mask) + mask*torch.randn_like(img)
    mask_img = img*(1. - mask) + mask

    cond_image_np = tensor2img(cond_image)

    # unsqueeze
    cond_image = cond_image.unsqueeze(0).to("cuda")
    gt_image = img.unsqueeze(0).to("cuda")
    mask = mask.unsqueeze(0).to("cuda")

    # inference
    with torch.no_grad():
        output, visuals = model.restoration(cond_image, y_t=cond_image,
                                            y_0=gt_image, mask=mask, sample_num=4)

    # output format 
    output_img = output.detach().float().cpu()
    resImg = tensor2img(output_img)

    return resImg, cond_image_np



def train_mask(filename, img_array, iter, lr, size, norm, y_lat=None, y_lng=None) :

    tv_beta = 3
    l1_coeff = size
    tv_coeff = norm
    batch_size=64
    num_workers=4

    im_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    intImg = cv2.resize(im_bgr, (256, 256))
    floatImg = np.float32(intImg) / 255
    varImg = preprocess_image(floatImg)

    gussianBlur, medianBlur, mixBlur = blurImg(intImg, floatImg)
    varBlur = preprocess_image(medianBlur)

    mask = initMask()# 生成mask

    # 获取真值
    # if its im2gps or random2k
    if isDatasetImage:
        y_lat, y_lng = getGV(groudValuePath, filename, 2, 3)

    if y_lat is not None:
        pre_class, pre_lat ,pre_lng, preds, _ = gps_inference(varImg, model) ## 12893个分类
        err = get_distance(pre_lat ,pre_lng, y_lat, y_lng)

        st.sidebar.write("prelat:", pre_lat)
        st.sidebar.write("prelng:", pre_lng)
        st.sidebar.write("error:", err)

        pre_class = pre_class.item()
        ground_pred = preds[0,pre_class]
	
    # 训练
    upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256)).cuda()
    optimizer = torch.optim.Adam([mask], lr=lr)

    sumLossList, sizeLossList, normLossList, classifyLossList, predList= [], [], [], [], []
    
    st.sidebar.text('Generating...')

    latest_iteration = st.sidebar.empty()
    bar = st.sidebar.progress(0)
    progress = 1

    for i in range(1, iter+1):
        if i % (iter/100) == 0:
            latest_iteration.text(f"{progress}%")
            bar.progress(progress)
            time.sleep(0.1)
            progress += 1
        
        upsampled_mask = upSample(upsample, mask)
        perturbated_input = varImg.mul(upsampled_mask) + varBlur.mul(1-upsampled_mask)

        noise = generateNoise()
        perturbated_input = perturbated_input + noise

        sumloss, sizeloss, normloss, classifyloss, pred, pre_lat, pre_lng, err = calLossWithCoarse(perturbated_input, mask, i, model, pre_class, y_lat, y_lng, tv_beta, l1_coeff, tv_coeff, printLog=False) 
        
        sumLossList.append(sumloss.item())
        sizeLossList.append(sizeloss.item())
        normLossList.append(normloss.item())
        classifyLossList.append(classifyloss.item())
        predList.append(pred.item())


        # 更新   
        optimizer.zero_grad() 
        sumloss.backward()
        optimizer.step()
        mask.data.clamp_(0, 1)

    lossDict = {
        "sumloss":sumLossList,
        "sizeloss":sizeLossList,
        "normloss":normLossList,
        "classifyloss":classifyLossList,
        "pred":predList
    }
    vis_result, floatImg, sharpMask, heatmap = resultVis("", mask, mixBlur, intImg, floatImg, upsample, "", str(i), False)

    return y_lat, y_lng, pre_lat.item(), pre_lng.item(), err, lossDict, vis_result, sharpMask, heatmap


## Main
st.title('Explain GeoEstimation')

# parameters
groudValuePath = "../resources/im2gps_places365.csv"
checkpoint="../models/base_M/epoch=014-val_loss=18.4833.ckpt"
hparams="base_M/hparams.yaml"

# sidebar:
# upload images
isDatasetImage = st.sidebar.checkbox('im2gps image')
uploaded_file = st.sidebar.file_uploader("Choose a image")

# hyperparameter adjusting
# iterations, lr, size = parameter_adust()
iterations = st.sidebar.slider('iterations', 500, 5000, 500, 1000)
lr = st.sidebar.select_slider("lr", options=[0.0001, 0.001, 0.005, 0.01, 0.1], value = 0.1)
size = st.sidebar.select_slider("size", options=[0.005, 0.01, 0.05, 0.1, 0.5], value = 0.01)
norm = st.sidebar.select_slider("norm", options=[0.05, 0.1, 0.2, 0.4, 0.6], value = 0.2)


parameters = {"iter":iterations, "lr":lr, "size":size, "norm":norm}

if uploaded_file is not None:
    image = load_img(uploaded_file)

    lat, lng =None, None
    if not isDatasetImage :
        lat, lng = getExifInfo(image)
        st.write("true lat:", lat)
        st.write("true lng:", lng)

    img_array = np.array(image)
    st.sidebar.image(
        image, caption='original image',
        # use_column_width=True
    )
    loss_button = st.sidebar.checkbox('loss plot')
    edit_check = st.sidebar.checkbox('edit check')
    generation_button = st.sidebar.button('Mask Generation')

    left_column, right_column = st.columns(2)

    if generation_button:

        # blackbox model 
        st.sidebar.text("Loading model...")
        model = load_model(checkpoint, hparams)
        st.sidebar.text("Done!")
        y_lat, y_lng, pre_lat, pre_lng, err, lossDict, vis_result, sharpMask, heatmap = train_mask(uploaded_file.name, img_array, parameters["iter"], parameters["lr"], parameters["size"], parameters["norm"],lat, lng)
        
        # palette model
        if edit_check :

            # load model 
            st.sidebar.text("Loading inpainting model...")
            with open ("base_M/paletteArgs.txt") as f :
                data = f.read()
            args = json.loads(data)
            model_pth = "../models/places/16_Network.pth"
            edit_model = load_edit_model(args, model_pth)
            st.sidebar.text("Done!")

            # load image
            baseImg = load_base_img(image)

            # inference
            editedImg, condImg = inpainting(baseImg, sharpMask, edit_model)        
        
        # result 
        left_column.image(
            image = cv2.cvtColor(np.uint8(255*vis_result), cv2.COLOR_BGR2RGB),          
            caption='masked image',
            use_column_width=True
        )
        right_column.image(
            image = cv2.cvtColor(np.uint8(255*heatmap), cv2.COLOR_BGR2RGB),
            caption="heatmap image",
            use_column_width=True
        )

        if edit_check :
            left_column.image(
                image = editedImg,          
                caption='edited image',
                use_column_width=True
            )
            right_column.image(
                image = condImg,
                caption="condition image",
                use_column_width=True
            )

        df = pd.DataFrame({
            "col1": [y_lat, pre_lat],
            "col2": [y_lng, pre_lng],
            "col3": ["#55ff00","#ff0000"],
        })

        left_column.map(df,
            latitude='col1',
            longitude='col2',
            color='col3',
            zoom = 2)
        
        with right_column :
            st.write("error distance:",err)

        if loss_button:
            st.scatter_chart(np.array(lossDict["sumloss"]))
            st.scatter_chart(np.array(lossDict["sizeloss"]))
            st.scatter_chart(np.array(lossDict["normloss"]))
            st.scatter_chart(np.array(lossDict["classifyloss"]))
            st.scatter_chart(np.array(lossDict["pred"]))

