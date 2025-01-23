import os
import streamlit as st
from PIL import Image
import io
from pipeline import Pipeline

def format_output(content):
    if content["typeID"] == "9so_front":
        result = {
            "typeID": content["typeID"],
            "id": content["id"],
            "name": content["name"],
            "birth_day": content["birth_day"],
            "origin_location": content["origin_location"],
            "recent_location": content["recent_location"]
        }
        
        return result
    elif content["typeID"] == "12so_front":
        result = {
            "typeID": content["typeID"],
            "id": content["id"],
            "name": content["name"],
            "gender": content["gender"],
            "birth_day": content["birth_day"],
            "expire_date": content["expire_date"],
            "nationality": content["nationality"],
            "origin_location": content["origin_location"],
            "recent_location": content["recent_location"]
        }
        return result
    else: 
        return content
    
yolo_best_path = "./weights/best.pt"
vietocr_model_config = "vgg_transformer"
vietocr_model_path = "./weights/vgg_transformer.pth"
device = "cpu"
# device = "cuda:0" # when using gpu

# pipeline = Pipeline(
#     model_detect_path=yolo_best_path,
#     model_text_recognition_config=vietocr_model_config,
#     model_text_recognition_path=vietocr_model_path,
#     device=device
# )

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = Pipeline(
        model_detect_path=yolo_best_path,
        model_text_recognition_config=vietocr_model_config,
        model_text_recognition_path=vietocr_model_path,
        device=device
    )

def predict_yolo_vietocr_id_card(image_path):
    result = st.session_state.pipeline(image_path)
    return result

UPLOAD_FOLDER = "./tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def save_image(image_file, upload_folder):
    file_path = os.path.join(upload_folder, image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.getbuffer())
    return file_path

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def local_js(file_name):
    with open(file_name) as f:
        st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("Identity Card Information Extraction")
local_css("Style.css")
local_js("Javascript.js")

col1, col2 = st.columns([1, 1])
st.markdown('<div class="container">', unsafe_allow_html=True)

with col1:
    st.markdown('<div class="col">', unsafe_allow_html=True)
    st.markdown('<div class="title"> Image </div>', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Chọn các file hình ảnh (PNG, JPG, JPEG)", type=["jpg", "png", "jpeg", "webp"], 
                                             label_visibility="collapsed",
                                             accept_multiple_files=False)

    cur_image_path = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        cur_image_path = save_image(uploaded_file, UPLOAD_FOLDER)
        st.image(image, use_container_width=True, caption=f"Image size: {image.size}")
    else:
        st.markdown('<div id="clickable-box" class="upload-box box_upload">No Images</div>', unsafe_allow_html=True)       
    st.markdown('</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="title"> Result </div>', unsafe_allow_html=True)

    extraction_dict = {
        "YOLO + VietOCR": predict_yolo_vietocr_id_card,
    }
    
    extraction_option = st.sidebar.selectbox("Etraction type: ", list(extraction_dict.keys()))
    st.sidebar.write("You selected:", extraction_option)
    extraction_func = extraction_dict[extraction_option] 

    if st.sidebar.button("SUBMIT"):
        result = extraction_func(cur_image_path)
        result = format_output(result)
        st.json(result)