from prometheus_client import Counter, Gauge, Histogram

CLASSIFY_REQUESTS = Counter(
    "deepcoin_classify_requests_total",
    "Total classification requests received",
    ["route_taken"]
)

CNN_CONFIDENCE = Gauge(
    "deepcoin_cnn_confidence",
    "Last recorded CNN confidence score"
)

HTTP_REQUEST_DURATION = Histogram(
    "deepcoin_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)
