from flask_restful import Api
from front import network
from front import app
from flask_cors import CORS
CORS(app, supports_credentials=True)

if __name__ == '__main__':
    # CORS(app,supports_credentials=True)
    api = Api(app)
    app.run(host='0.0.0.0', port=5001, debug=False)
    # cfg = Config.fromfile('/home/chase/PycharmProjects/MMFeDServer/front/faster_rcnn_r50_fpn_2x_WiderFace_FedDGA.py')
    # cfg.model.rpn_head.pop(0)
    # print(cfg.model.rpn_head)
    # print(1)