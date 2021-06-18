from flask import Flask
from controller.controller import segmentation_controller
from werkzeug.middleware.proxy_fix import ProxyFix
import controller.response as response
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)

app = Flask(__name__)
app.register_blueprint(blueprint=segmentation_controller)


@app.route(rule="/")
def default_page():
    return response.response_success("ok")

def main():
    app.run(host="0.0.0.0", port=1488)


app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
    main()
