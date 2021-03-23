from flask import request, jsonify, Flask
from flask_restful import abort, Resource, Api 
from srn import config, srn

app = Flask(__name__)
api = Api(app)


pose_predictor = srn.handpose

class Pipeline(Resource):

    def post(self):
        """
        Receive request and provide response
        """
        data = request.get_json(force=True)
        poses = pose_predictor.run(data["data_dir"])
        # print(poses[0][0])
        return {"message": "Success"}

api.add_resource(Pipeline, "/")

if __name__ == "__main__":
    app.run(debug=True, port=9000)