from flask import Flask, jsonify, request, json
from flask_cors import CORS, cross_origin
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'Images/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

jsonFile = open('songdata.json')
jsonAllFile = open('allsongs.json')

jsonData = json.load(jsonFile)
jsonAllData = json.load(jsonAllFile)

songs = []
allSongs = []
# load json data from json file
for item in jsonData:
    songs.append(item)
# load all songs json data from json file
for song in jsonAllData:
    allSongs.append(song)


@app.route("/", methods=['GET'])
def hello_world():
    return 'Emoscape backend!'


# get all the songs
@app.route('/song')
@cross_origin(origin='*')
def get_all_playlists():
    return jsonify(allSongs)


# get all the songs from the playlist of each mood
@app.route('/song/<string:mood>')
@cross_origin(origin='*')
def get_mood_playlist(mood):
    for song in songs:
        if song['mood'] == mood:
            return jsonify(song)
    return jsonify({'message': 'store not found'})


# add new songs to specific playlist
@cross_origin(origin='*')
@app.route('/song/<string:mood>/song', methods=['POST'])
def add_songs(mood):
    request_data = request.get_json()
    for song in songs:
        if song['mood'] == mood:
            new_item = {
                'title': request_data['title'],
                'artist': request_data['artist'],
                'link': request_data['link'],
                'id': request_data['id'],
                'genre': request_data['genre'],
                'album': request_data['album'],
                'imageurl': request_data['imageurl']
            }
            song['song'].append(new_item)
            return jsonify(new_item)
    return jsonify({'message': 'store not found'})


@app.route('/song', methods=['POST'])
@cross_origin(origin='*')
def create_newPlaylist():
    request_data = request.get_json()
    new_store = {
        'mood': request_data['mood'],
        'song': []
    }
    songs.append(new_store)
    return jsonify(new_store)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# method to predict the emotion
@app.route('/image', methods=['GET', 'POST'])
def predict_emotion():
    model = load_model('Model/trained_model_1.h5')

    class_labels = ['angry', 'happy', 'sad', 'surprise', 'neutral']

    if request.method == 'GET':
        f = open("mood.txt", "r")

        return jsonify({'mood': f.read()})

    if request.method == 'POST':

        if 'file' not in request.files:
            print('No file part')
            return jsonify({'message': 'not found'})

        print("Posted file: {}".format(request.files['file']))
        file = request.files['file']

        if file.filename == '':
            print('No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            base_dir = r'Images'
            os.path.join(base_dir, filename)
            img = image.load_img(os.path.join(base_dir, filename), color_mode="grayscale", target_size=(48, 48))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x /= 255

            custom = model.predict(x)
            label = class_labels[custom.argmax()]

            f = open("mood.txt", "w")
            f.write(label)
            f.close()

            return jsonify({'mood': label})


if __name__ == '__main__':
    app.run(debug=True)
