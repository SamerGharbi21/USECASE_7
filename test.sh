#!/bin/bash
curl -X POST "https://usecase-7.onrender.com/predict" -H "Content-Type: application/json" \
-d '{"age":23 , "current_value":5000000000}'
