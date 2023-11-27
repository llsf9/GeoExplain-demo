import streamlit as st
from PIL import Image
import time
import numpy as np
from utils import * 
from ImgProcess import * 
from model import * 
from visualize import *

@st.cache_resource
def load_model(checkpoint, hparams) :
    model = loadModel(checkpoint, hparams)
    return model

@st.cache_data
def load_img(uploaded_file) :
    image = Image.open(uploaded_file)
    return image 


def train_mask(filename, img_array, iter, lr, size) :

    tv_beta = 3
    l1_coeff = size
    tv_coeff = 0.3
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
    y_lat, y_lng = getGV(groudValuePath, filename, 2, 3)
    re_lat, re_lng, prob = gps_inference(varImg, model) ## 12893个分类
    err = get_distance(re_lat,re_lng, y_lat, y_lng)
    ground_value = prob.argmax().item()
    ground_prob = prob[0,ground_value]
	
    # 训练
    upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256)).cuda()
    optimizer = torch.optim.Adam([mask], lr=lr)
    
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

        loss, pre_lat, pre_lng, err = calLoss2(perturbated_input, mask, i, model, ground_value, y_lat, y_lng, tv_beta, l1_coeff, tv_coeff, printLog=False) 


        # 更新   
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        mask.data.clamp_(0, 1)

    vis_result, floatImg, sharpMask, heatmap = resultVis("", mask, mixBlur, intImg, floatImg, upsample, "", str(i), False)

    return y_lat, y_lng, pre_lat.item(), pre_lng.item(), err, vis_result, heatmap


## Main
st.title('Explain GeoEstimation')

# parameters
groudValuePath = "../resources/im2gps_places365.csv"
checkpoint="../models/base_M/epoch=014-val_loss=18.4833.ckpt"
hparams="base_M/hparams.yaml"

# sidebar:
# upload images
st.sidebar.text("Loading model...")
model = load_model(checkpoint, hparams)
st.sidebar.text("Done!")
uploaded_file = st.sidebar.file_uploader("Choose a image")

# hyperparameter adjusting
# iterations, lr, size = parameter_adust()
iterations = st.sidebar.slider('iterations', 1000, 5000, 1000, 1000)
lr = st.sidebar.select_slider("lr", options=[0.0001, 0.001, 0.005, 0.01, 0.1], value = 0.1)
size = st.sidebar.select_slider("size", options=[0.005, 0.01, 0.05, 0.1, 0.5], value = 0.01)

parameters = {"iter":iterations, "lr":lr, "size":size}

if uploaded_file is not None:
    image = load_img(uploaded_file)
    img_array = np.array(image)
    st.sidebar.image(
        image, caption='original image',
        # use_column_width=True
    )
    generation_button = st.sidebar.button('Mask Generation')
    # im_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # intImg = cv2.resize(im_bgr, (256, 256))
    # floatImg = np.float32(intImg) / 255

    # varImg = preprocess_image(floatImg)
    # re_lat, re_lng, prob = gps_inference(varImg, model)
    # "re_lat", re_lat
    # "re_lng", re_lng

    left_column, right_column = st.columns(2)

    if generation_button:
        y_lat, y_lng, pre_lat, pre_lng, err, vis_result, heatmap = train_mask(uploaded_file.name, img_array, parameters["iter"], parameters["lr"], parameters["size"])
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

        df = pd.DataFrame({
            "col1": [y_lat, pre_lat],
            "col2": [y_lng, pre_lng],
            "col3": ["#55ff00","#ff0000"],
        })

        left_column.map(df,
            latitude='col1',
            longitude='col2',
            color='col3',
            zoom = 1)
        
        with right_column :
            st.write("error distance:",err)