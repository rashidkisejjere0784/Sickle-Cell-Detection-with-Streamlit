import streamlit as st
import numpy as np
import torch
from torch import nn
from io import BytesIO
from PIL import Image

device = "cpu"

class DCNNSCD(nn.Module):
    def __init__(self,input_channels : int, output_shape : int):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels= input_channels,
                out_channels=32,
                kernel_size = 2,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels=64,
                kernel_size = 2,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels= 64,
                kernel_size = 2,
                stride = 1,
                padding = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.output_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=50176, out_features=64),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 64, out_features=32),
            nn.Linear(in_features=32, out_features=1)
        )
        
    def forward(self, x):
        x = self.cnn_block_1(x)
        
        x = self.cnn_block_2(x)
        
        x = self.cnn_block_3(x)
        
        x = self.output_block(x)
        
        return x
    
    
model = DCNNSCD(input_channels=3, output_shape=1).to(device)

model.load_state_dict(torch.load("model state dict.pt", map_location=torch.device('cpu')))
model.to(device)

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to 250x250 pixels
    image = image.resize((224, 224))

    image_array = np.array(image)
    
    # Normalize the image to the range [0, 1]
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array.astype(np.float32), axis=0)
    print(image_array.shape)
    
    image_tensor = torch.from_numpy(image_array).to(device)
    image_tensor = image_tensor.permute(0, 3, 1, 2)

    return image_tensor

# Streamlit application
st.title("Sickle Cell Detection")

st.write("""
    This application allows you to upload a blood smear image, which will be analyzed to detect the absence of sickle cells.
    Please upload an image below.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    # Open the image using PIL
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing...")
    processed_image = preprocess_image(image)
    

    model.eval()
    with torch.inference_mode():
        print(processed_image.shape)
        prediction = model(processed_image)
        pred_class = torch.round(torch.sigmoid(prediction))
    
    if pred_class == 0:
        st.success("The model predicts that there are no sickle cells in the uploaded image.")
    else:
        st.error("The model predicts that there might be sickle cells in the uploaded image.")

# Additional Information or Footer
st.sidebar.title("About")
st.sidebar.info("""
    This Sickle Cell Detection application uses a deep CNN for Sickle Cell Detection (DCNNSCD), reach to us for more information at [rashidkisejjere0784@gmail.com](mailto:rashidkisejjere0784@gmail.com).
""")
