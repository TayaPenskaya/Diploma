#!/bin/bash

gunicorn --bind=0.0.0.0:5488 --preload --reload \
    --log-level info init:app
