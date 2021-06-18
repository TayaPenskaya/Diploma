from main import Segmentation
import json
import controller.response as response
from flask import Blueprint, request

segmentation_controller = Blueprint(name="get_segmentation",
                                    import_name=__name__,
                                    url_prefix="/get_segmentation")

segm = Segmentation()

@segmentation_controller.route(rule="/", methods=["POST"])
def get_segmentation():
    image = request.form["image"]

    image_segmented = segm.get_segmentation(image)

    return response.response_success(image_segmented)
