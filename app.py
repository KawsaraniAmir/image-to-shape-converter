import random
import numpy as np
from PIL import Image, ImageDraw
import threading
import os
import io
from flask import Flask, render_template, request, jsonify, send_file
from time import sleep
import cv2

app = Flask(__name__)

# Path to save canvas and uploaded files
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# In-memory canvas
canvas_image = None
stop_event = threading.Event()

# Global variables to track the similarity and scale changes
best_similarity = -1  # Initially set to a very low value
initial_scale = 1.0
scale_decrease_rate = 0.98  # Factor to reduce scale


def load_original_image():
    global original_image
    original_image = Image.open("uploads/uploaded_image.png")


def calculate_positive_pixel_change(img1, img2, threshold=10):
    # Ensure both images are in the same mode (e.g., RGB or RGBA)
    img1 = img1.convert('RGBA')
    img2 = img2.convert('RGBA')

    # Converting images to numpy arrays for pixel comparison
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Compute the absolute difference between corresponding pixels
    diff = np.abs(img1_array - img2_array)

    # Count how many pixels have a difference smaller than the threshold (positive change)
    positive_changes = np.sum(np.all(diff <= threshold, axis=-1))  # Count pixels that match or improve
    return positive_changes


# Funzione per calcolare la somiglianza tra l'immagine originale e quella modificata
def calculate_similarity(img1, img2):
    return calculate_positive_pixel_change(img1, img2)


def get_average_color(image, x1, y1, x2, y2):
    # Assicurati che la regione sia valida e non fuori dai limiti dell'immagine
    width, height = image.size
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, width), min(y2, height)

    if x1 >= x2 or y1 >= y2:  # Se le coordinate non formano una regione valida
        return (128, 128, 128)  # Colore grigio neutro come fallback

    # Estrai la regione dall'immagine
    region = image.crop((x1, y1, x2, y2))

    # Converti la regione in un array NumPy
    region_array = np.array(region)

    # Calcola la media sui canali (R, G, B) della regione
    avg_color = region_array.mean(axis=(0, 1))

    # Se la media è NaN (ad esempio, se la regione è completamente nera o trasparente), restituisci un colore di fallback
    if np.isnan(avg_color).any():
        return (128, 128, 128)  # Colore grigio neutro come fallback

    # Restituisci il colore medio come tupla (R, G, B)
    return tuple(map(int, avg_color))  # Restituisce il colore medio


# Funzione per generare una forma casuale
def generate_random_shape(canvas_image):
    width, height = canvas_image.size
    x1, y1 = random.randint(0, width), random.randint(0, height)
    x2, y2 = random.randint(0, width), random.randint(0, height)

    # Ensure x2 >= x1 and y2 >= y1
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    color = get_average_color(original_image, x1, y1, x2, y2)
    draw = ImageDraw.Draw(canvas_image)
    shape_type = random.choice(['circle', 'rectangle', 'line'])
    
    if shape_type == 'circle':
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif shape_type == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], fill=color)
    else:  # Line
        draw.line([x1, y1, x2, y2], fill=color, width=2)

    return canvas_image, (x1, y1, x2, y2, shape_type, color)


# Funzione per applicare trasformazioni (rotazione o ridimensionamento)
def apply_transformation(shape_details, canvas_image, transform_type='rotate', scale_factor=1.0):
    x1, y1, x2, y2, shape_type, color = shape_details
    if transform_type == 'rotate':
        angle = random.uniform(-10, 10)
        canvas_image = canvas_image.rotate(angle, resample=Image.BICUBIC, center=((x1 + x2) // 2, (y1 + y2) // 2))
    elif transform_type == 'resize':
        width_new = int((x2 - x1) * scale_factor)
        height_new = int((y2 - y1) * scale_factor)
        x2 = x1 + width_new
        y2 = y1 + height_new
    return canvas_image


# Funzione ottimizzata per generare forme e selezionare le migliori
def generate_and_select_best_shape(original_image, canvas_image):
    best_shapes = []
    start_amount = 100  # Numero di forme da generare
    iterations = 5  # Numero di iterazioni per selezionare le migliori forme

    global initial_scale  # Traccia la scala iniziale

    # 1. Genera forme casuali e calcola la somiglianza
    for _ in range(start_amount):
        temp_canvas, shape_details = generate_random_shape(canvas_image.copy())
        similarity_score = calculate_similarity(temp_canvas, original_image)
        best_shapes.append((similarity_score, shape_details))

    # 2. Ordina le forme in base alla somiglianza e seleziona le 10 migliori
    best_shapes = sorted(best_shapes, reverse=True, key=lambda x: x[0])
    best_shapes = best_shapes[:10]  # Seleziona le prime 10 forme migliori

    # 3. Applica trasformazioni alle forme migliori
    next_best_shapes = []
    for _, shape_details in best_shapes:
        temp_canvas = canvas_image.copy()
        for i in range(iterations):
            # Cambia la scala globalmente per tutte le forme
            scale_factor = initial_scale  # Scala che cambia progressivamente
            temp_canvas = apply_transformation(shape_details, temp_canvas, transform_type='resize', scale_factor=scale_factor)
            similarity_score = calculate_similarity(temp_canvas, original_image)
            next_best_shapes.append((similarity_score, shape_details))

    # 4. Ordina le nuove forme trasformate per somiglianza
    next_best_shapes = sorted(next_best_shapes, reverse=True, key=lambda x: x[0])
    final_best_shapes = next_best_shapes[:10]  # Seleziona le 10 migliori forme trasformate

    # 5. Seleziona la miglior forma finale
    final_shape_details = final_best_shapes[0][1]
    best_score = final_best_shapes[0][0]


    print("Best score: " + str(best_score))  # Converte best_score in stringa prima di concatenarlo
    print("Current scaling: " + str(initial_scale))  # Converte initial_scale in stringa prima di concatenarlo
    print('Max similarity: '+ str(best_similarity))

    return final_shape_details


def generate_shapes():
    global canvas_image
    while not stop_event.is_set():
        # Select the best shape after transformations
        final_shape_details = generate_and_select_best_shape(original_image, canvas_image)
        x1, y1, x2, y2, shape_type, color = final_shape_details
        draw = ImageDraw.Draw(canvas_image)
        if shape_type == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape_type == 'line':  # Line
            draw.line([x1, y1, x2, y2], fill=color, width=2)

        # Sleep for a short period to reduce CPU usage
        sleep(0.05)


# Route to fetch the updated canvas
@app.route('/canvas')
def canvas():
    global canvas_image
    img_io = io.BytesIO()
    canvas_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


# Route to stop the shape generation process
@app.route('/', methods=['GET', 'POST'])
def index():
    global canvas_image
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded", 400

        image = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
        image.save(image_path)

        # Load the original image once
        load_original_image()

        # Create a new canvas
        with Image.open(image_path) as img:
            canvas_image = Image.new("RGB", img.size, (255, 255, 255))  # Blank canvas

        # Start shape generation in the background
        stop_event.clear()
        threading.Thread(target=generate_shapes, daemon=True).start()

        return jsonify({'status': 'started'})

    return render_template('shapegen.html')


if __name__ == '__main__':
    app.run(debug=True)
