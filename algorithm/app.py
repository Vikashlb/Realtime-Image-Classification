from flask import Flask, render_template, request
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the path to a temporary directory where uploaded files will be saved
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the UPLOAD_FOLDER exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
loaded_model = tf.keras.models.load_model("C:/Users/lbvik/Desktop/MiniProject/EditedDataSetH3.h5")

# Fit the LabelEncoder with the class labels
output_classes = ["eye", "foot", "nose", "ear", "hand"]
label_encoder = LabelEncoder()
label_encoder.fit(output_classes)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the POST request has a file part
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return "No selected file"

        if file:
            # Ensure the filename is secure to prevent malicious file execution
            filename = secure_filename(file.filename)

            # Save the uploaded file to a temporary directory
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Load and preprocess the input image
            input_img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224))
            input_img = image.img_to_array(input_img)
            input_img = np.expand_dims(input_img, axis=0)
            input_img = tf.keras.applications.efficientnet.preprocess_input(input_img)

            # Use the trained model to make a prediction
            prediction = loaded_model.predict(input_img)

            # Map the prediction to the corresponding category label
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            return render_template("index.html", image_uploaded=True, prediction=predicted_label)

    return render_template("index.html", image_uploaded=False)

if __name__ == "__main__":
    app.run(debug=True)
#http://127.0.0.1:5000/