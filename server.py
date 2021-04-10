from flask import request, jsonify, Flask
from flask_restful import abort, Resource, Api 
import pickle
import configparser
import os

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
        label = gesture_recognizer.predict(model_input)
        gif_path = os.getcwd() + "/srn/results/action.gif"
        return {
            "message": label,
            "gif_path": gif_path
        }

api.add_resource(Pipeline, "/")

#######
# Run #

if __name__ == "__main__":
    print(C['server'])
    app.run(debug=True, port=C['server']['port'])

