import random
import time

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

CLASS_NAMES = [
    "goldfish",
    "great_white_shark",
    "tiger_shark",
    "hammerhead",
    "electric_ray",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house_finch",
    "junco",
    "indigo_bunting",
    "robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water_ouzel",
    "kite",
    "bald_eagle",
    "vulture",
    "great_grey_owl",
    "fire_salamander",
    "smooth_newt",
    "newt",
    "spotted_salamander",
    "axolotl",
    "bullfrog",
    "tree_frog",
    "tailed_frog",
    "loggerhead",
    "leatherback_turtle",
    "mud_turtle",
    "terrapin",
    "box_turtle",
    "banded_gecko",
    "common_iguana",
    "american_chameleon",
    "whiptail",
    "agama",
]

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Number of predictions per class",
    ["class_name"],
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
CONFIDENCE_SCORE = Histogram(
    "confidence_score",
    "Predicted confidence score (softmax probability)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

app = FastAPI(title="ML Model Monitoring Demo")
Instrumentator().instrument(app).expose(app)

metrics_app = make_asgi_app()
app.mount("/metrics-raw", metrics_app)


class PredictRequest(BaseModel):
    image_id: str = "sample"


class PredictResponse(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    inference_time_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.perf_counter()

    time.sleep(random.uniform(0.002, 0.015))

    class_idx = random.randint(0, len(CLASS_NAMES) - 1)
    class_name = CLASS_NAMES[class_idx]
    confidence = random.betavariate(5, 2)

    elapsed = time.perf_counter() - start

    PREDICTIONS_TOTAL.labels(class_name=class_name).inc()
    INFERENCE_LATENCY.observe(elapsed)
    CONFIDENCE_SCORE.observe(confidence)

    return PredictResponse(
        class_name=class_name,
        class_index=class_idx,
        confidence=round(confidence, 4),
        inference_time_ms=round(elapsed * 1000, 2),
    )
