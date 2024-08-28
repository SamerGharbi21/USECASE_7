#!/bin/bash
curl -X POST "https://api-nvaa.onrender.com/predict" -H "Content-Type: application/json" \
-d '{"age":30 , "current_value":1000000000}'
