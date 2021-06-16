#!/bin/bash

gunicorn --bind=0.0.0.0:1488 --preload --reload \
    --log-level info init:app
