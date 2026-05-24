import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
import MLP_EMNIST as mlp
import numpy as np
import cv2
import io
from PIL import Image

st.set_page_config(page_title="Identificador de Letras e Números", page_icon=":robot_face:", layout="centered")
st.title("Identificador de Letras e Números", text_alignment = 'center')
st.markdown("Este projeto utiliza uma **rede neural** para identificar letras e números a partir do conjunto de dados **EMNIST**. O modelo é treinado para classificar imagens de caracteres em **47 classes** diferentes, incluindo letras maiúsculas, minúsculas e dígitos numéricos.")

#st.divider()

# Mapeamento oficial do EMNIST Balanced (47 classes)
class_mapping = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def preprocess_canvas_image(img):
    # img vem como RGBA (H, W, 4)
    
    # 1. Converter para grayscale
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    
    # 2. Inverter (fundo branco -> preto)
    img = 255 - img
    
    # 3. Threshold (binarizar)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # 4. Encontrar bounding box
    coords = cv2.findNonZero(img)
    
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    
    # 5. Redimensionar mantendo proporção
    max_side = max(w, h)
    square = np.zeros((max_side, max_side), dtype=np.uint8)
    
    # centralizar no quadrado
    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    # 6. Resize para 28x28
    img = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 7. APLICAR TRANSFORMAÇÕES DO TREINO
    img = cv2.flip(img, 1)  # flip horizontal
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 90° esquerda
    
    # 8. Normalizar igual EMNIST
    img = img / 255.0
    img = (img - 0.2860) / 0.3530
    
    # 9. Flatten
    img = torch.tensor(img, dtype=torch.float32).view(-1)
    
    return img


model = mlp.MLP_EMNIST()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(r"backup_models\best model,.pth", weights_only=True, map_location=device)
model.load_state_dict(checkpoint['modelo_state'])
model.to(device)
model.eval()

tab_canvas, tab_upload_image = st.tabs(["✍️ Canvas", "🖼️ Upload de Imagem"], key="tabs", on_change='rerun')

with tab_canvas:
    col1, col2, col3 = st.columns(3, gap = 'xlarge', vertical_alignment='center')

    with col1:
        with st.container(horizontal_alignment='center'):
            stroke_color = st.color_picker("Cor do traço", "#000000", key="stroke_color")

    with col2:
        with st.container(horizontal_alignment='center'):
            stroke_width = st.slider("Espessura do traço", 10, 40, 18, key="stroke_width")

    with col3:
        with st.container(horizontal_alignment='center'):
            bg_color = st.color_picker("Cor de fundo", "#FFFFFF", key="bg_color")

    # Canvas principal - mais bonito e com melhores configurações
    canvas_result = st_canvas(
        drawing_mode="freedraw",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=None,           # pode adicionar imagem depois
        height=450,
        width=700,
        display_toolbar=True,
        update_streamlit=True,
        key="main_canvas",
        # Estilo extra
        initial_drawing=None
    )


with tab_upload_image:
    image_uploaded = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"], help="Tamanho máximo: 50MB", key="image_upload", max_upload_size=50)
    
    st.write(image_uploaded)


if st.session_state["tabs"] == "🖼️ Upload de Imagem":
    if image_uploaded is not None:
        img = np.array(Image.open(io.BytesIO(image_uploaded.read())))
        
    else:
        st.markdown(
            f"""
            <div style="
                background-color: #1f2937;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
            ">
                <h2>Adicione uma imagem!</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("#### Nível de confiança")
        st.progress(float(0), f"{float(0)*100:.2f}%")
        st.stop()

else:
    if canvas_result is not None:
        if len(canvas_result.json_data['objects']) != 0: 
            img = canvas_result.image_data
        
        else:
            st.markdown(
                f"""
                <div style="
                    background-color: #1f2937;
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                ">
                    <h2>Adicione uma imagem!</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("#### Nível de confiança")
            st.progress(float(0), f"{float(0)*100:.2f}%")
            st.stop()
    
    else:
        st.markdown(
            f"""
            <div style="
                background-color: #1f2937;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
            ">
                <h2>Adicione uma imagem!</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("#### Nível de confiança")
        st.progress(float(0), f"{float(0)*100:.2f}%")
        st.stop()


with torch.no_grad():
    # Processar a imagem do canvas
    img = preprocess_canvas_image(img)  # Adiciona batch dimension e move para o dispositivo
    
    if img is None:
        st.write("Nenhum traço detectado. Por favor, desenhe algo no canvas.")
        st.stop()
    # Fazer a previsão
    
    img = img.unsqueeze(0).to(device)
    
    with st.spinner("Analisando imagem..."):
        output = model(img)
    
        # Obter probabilidades usando Softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confianca, pred_idx = torch.max(probabilities, 1)
        
        caractere_previsto = class_mapping[pred_idx.item()]

st.markdown("---")

st.markdown(
    f"""
    <div style="
        background-color: #1f2937;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
    ">
        <h2>Previsão do Modelo</h2>
        <h1 style="font-size: 60px; margin: 10px 0;">{caractere_previsto}</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("#### Nível de confiança")
st.progress(float(confianca[0]), f"{float(confianca[0])*100:.2f}%")
