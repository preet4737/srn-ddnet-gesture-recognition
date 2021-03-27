from flask import request, jsonify, Flask
from flask_restful import abort, Resource, Api 
import pickle
import configparser

from srn import config, HandPose
from ddnet.sampling import sampling_frame
from ddnet import Predictor

#################
# Configuration #

C = configparser.ConfigParser()
C.read(['server.cfg', 'server.local.cfg'])

#
app = Flask(
        __name__,
        static_folder=C['dataset']['path'],
        static_url_path='/data'
        )
app.url_map.strict_slashes = False
#
api = Api(app)
#

#######################
# Dataset Information #

with open( C['dataset']['index_file'], 'rb' ) as f:
    index = pickle.load( f )

#

@app.route('/info/<g>')
def get_essais( g: str ) -> list:
    """ Returns essai list for a gesture """
    essais = index[g]['essais']
    essais = jsonify(essais)
    return essais

@app.route('/info/<g>/<f>/<s>/<e>')
def get_pcds( g: str, f: str, s: str, e: str ) -> list:
    """ Returns PCD list for an essai """
    pcds = index[g][f][s][e]
    pcds = jsonify(pcds)
    return pcds

####################
# Machine Learning #

pose_predictor = HandPose(config)
gesture_recognizer = Predictor()

class Pipeline(Resource):
    def get(self):
        return {
            "urls": ["/info/", "/data/", "/"]
        }

    def post(self):
        """
        Receive request and provide response
        """
        data = request.get_json(force=True)
        poses = pose_predictor.run(data["data_dir"])
        model_input = sampling_frame(poses)
        label = gesture_recognizer.predict(model_input) # Add new model
        print(label)
        return {"message": "Success"}

api.add_resource(Pipeline, "/")

#######
# Run #

if __name__ == "__main__":
    print(C['server'])
    app.run(debug=True, port=C['server']['port'])

