from modules import breedDetection, colorDetection
from flask import *
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("Index.html")

@app.route("/color")
def get_color():
    filename = request.args.get("filename")
    predominant_color = colorDetection.ColorDetection.get_predominant_color(filename=filename)
    hex_color = colorDetection.ColorDetection.rgb_to_hex(predominant_color)
    return jsonify({"color": hex_color})

@app.route("/result",methods = ['POST'])
def classify_image():
    if request.method == 'POST':

        uploaded_file = request.files['image']
        animal = request.form['animal']

        filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(os.path.join('static/images', filename))
        #print(filename,uploaded_file)
        
        breed, probability = breedDetection.BreedModel.classify_image(classifier=animal,filename=filename)
        predominant_color = colorDetection.ColorDetection.get_predominant_color(filename=filename)
        color_name = colorDetection.ColorDetection.get_color_name(predominant_color)

        return render_template("Result.html", name = breed, filename = filename, probability=probability, color=color_name, hex_color='hex_color') 

if __name__ == "__main__":
	app.run(debug=True)