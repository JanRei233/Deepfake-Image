#!/bin/bash
cd backend
uvicorn deepfake:app --host 0.0.0.0 --port $PORT
