from flask import jsonify
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)


def response_internal_server_error(error: Exception) -> (str, int):
    logger.exception(error)
    return jsonify({"body": str(error)}), 500


def response_success(body):
    return jsonify({"body": body}), 200


def response_fail_auth():
    return jsonify({"body": "Unauthorized"}), 401


def response_not_found(body=None):
    return jsonify({"body": body}) if body is not None else jsonify({"body": "File not found"}), 404

def response_custom(body, code):
    return jsonify({"body": body}), code

