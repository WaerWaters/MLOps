#!/usr/bin/env bash
# Fires 200 requests at the /predict endpoint to populate Grafana metrics.
echo "Sending 2000 requests to http://localhost:8000/predict ..."
for i in $(seq 1 2000); do
    curl -s -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d "{\"image_id\": \"img_$i\"}" > /dev/null
done
echo "Done. Open Grafana at http://localhost:3000"
