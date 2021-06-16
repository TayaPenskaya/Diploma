from main import Text2Pose

import json
import controller.response as response
from flask import Blueprint, request

text2pose_controller = Blueprint(name="get_pose",
                                 import_name=__name__,
                                 url_prefix="/get_pose")

text2pose = Text2Pose()

@text2pose_controller.route(rule="/", methods=["POST"])
def get_pose():
    text = request.form["text"]

    pose = text2pose.get_pose_by_text(text)

    return response.response_success(pose)
