from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageDraw, ImageFont
import torch
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
# from Tools.metrics import MacroMetrics
import os

def process_fashion_image(image_filename):
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models (only once - could be moved outside function for better performance)
    gender_model = load_model(os.path.join("Output_Models", "gender_classification_CNN", "best_model.keras"), compile=False)
    articletype_model = load_model(os.path.join("Output_Models", "articleType_classification_CNN", "best_model.keras"), compile=False)
    basecolour_model = load_model(os.path.join("Output_Models", "baseColour_classification_CNN", "best_model.keras"), compile=False)
    
    # BLIP model initialization
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    # Class labels
    gender_class = ['Boys', 'Girls', 'Men', 'Unisex', 'Women']
    articletype_class = ['Backpacks', 'Belts', 'Bra', 'Briefs', 'Caps', 'Casual Shoes', 'Clutches', 
                        'Deodorant', 'Dresses', 'Earrings', 'Flats', 'Flip Flops', 'Formal Shoes', 
                        'Handbags', 'Heels', 'Innerwear Vests', 'Jackets', 'Jeans', 'Kurtas', 
                        'Kurtis', 'Lipstick', 'Nail Polish', 'Perfume and Body Mist', 'Sandals', 
                        'Sarees', 'Shirts', 'Shorts', 'Socks', 'Sports Shoes', 'Sunglasses', 
                        'Sweaters', 'Sweatshirts', 'Ties', 'Tops', 'Track Pants', 'Trousers', 
                        'Tshirts', 'Tunics', 'Wallets', 'Watches']
    basecolour_class = ['Black', 'Blue', 'Brown', 'Gold', 'Green', 'Grey', 'Multi', 'Orange', 
                       'Pink', 'Purple', 'Red', 'Silver', 'White', 'Yellow']
    
    # Process image
    new_image_path = os.path.join("Data", "resized_images", image_filename)
    processed_img = preprocess_image(new_image_path)
    
    # Make predictions
    gender_prediction = gender_model.predict(processed_img)
    articletype_prediction = articletype_model.predict(processed_img)
    basecolour_prediction = basecolour_model.predict(processed_img)
    
    # Get class names
    name_gender = gender_class[np.argmax(gender_prediction[0])]
    name_articletype = articletype_class[np.argmax(articletype_prediction[0])]
    name_basecolour = basecolour_class[np.argmax(basecolour_prediction[0])]
    
    print(f'articletype: {name_articletype}\nbasecolour: {name_basecolour}\ngender: {name_gender}')
    
    # Generate caption
    # prompt_text = f'A {name_basecolour.lower()} {name_gender.lower()} {name_articletype.lower()}'
    # prompt_text = f'{name_articletype} {name_gender} {name_basecolour}'
    prompt_text = f"A stylish {name_basecolour.lower()} {name_gender.lower()} {name_articletype.lower()} from "


    caption = generate_caption_with_prompt(new_image_path, prompt_text, processor, model, device)
    
    print("Generated Caption:", caption)
    
    
    return {
        'gender': name_gender,
        'article_type': name_articletype,
        'base_colour': name_basecolour,
        'caption': caption,
        'image_path': new_image_path
    }

# Helper functions
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size) 
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_caption_with_prompt(image_path, prompt_text, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=15)
    return processor.decode(out[0], skip_special_tokens=True)


def add_info_to_image(original_image_path, info_dict, output_path=None):

    img = Image.open(original_image_path)
    width, height = img.size
    

    new_height = height + 200 
    new_img = Image.new('RGB', (width, new_height), color='white')
    
    new_img.paste(img, (0, 150))    

    draw = ImageDraw.Draw(new_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    y_position = 10
    line_height = 45
    
    info_texts = [
        f"Gender: {info_dict['gender']}",
        f"Article Type: {info_dict['article_type']}",
        f"Base Colour: {info_dict['base_colour']}"
    ]
    
    for text in info_texts:
        draw.text((10, y_position), text, fill="black", font=font)
        y_position += line_height
    
    caption = info_dict['caption']
    if len(caption) > 50: 
        parts = [caption[i:i+50] for i in range(0, len(caption), 50)]
        for part in parts:
            draw.text((10, y_position), part, fill="black", font=font)
            y_position += line_height
    else:
        draw.text((10, y_position), caption, fill="black", font=font)
    
    if output_path:
        new_img.save(output_path)
    else:
        new_img.show()
    
    return new_img


img = "49321.jpg"
result = process_fashion_image(img)
print(result)

output_image_path = os.path.join("Data", "annotated_images", img)

annotated_img = add_info_to_image(result['image_path'], result, output_image_path)
annotated_img.show()