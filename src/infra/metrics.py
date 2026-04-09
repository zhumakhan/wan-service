from enum import Enum

from prometheus_client import Gauge, Counter, start_http_server, Histogram


class RequestStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    IN_PROGRESS = "in_progress"


METRICS_PREFIX = "wan_image"


class MetricsCollector:

    PORT = 8000

    def __init__(self):
        self.server, self.t = start_http_server(self.PORT)
        self.total_request_counter = Counter("request_count", f"Total request count", labelnames=["lora", "status"])
        self.in_progress_request_counter = Gauge("in_progress_request_count", f"In progress request count", labelnames=["lora"])
        self.inference_duration = Histogram("inference_duration", f"Inference duration", buckets=[25, 65, 90, 300])
        self.warmup_duration = Gauge("warmup_duration", f"Warmup duration")
        self.queue_wait_time = Counter("queue_wait_time", f"Time the message waits in the queue before processing")
        self.total_work_duration = Histogram("total_work_duration", f"Total work duration", buckets=[35, 75, 100, 650]) # p50, p90, p99, p99.9

    def record_request(self, lora: str, status: RequestStatus, value: int = 1):
        if status == RequestStatus.IN_PROGRESS:
            if value > 0:
                self.in_progress_request_counter.labels(lora=lora).inc(value)
            else:
                self.in_progress_request_counter.labels(lora=lora).dec(-value)
        else:
            self.total_request_counter.labels(lora=lora, status=status.value).inc(value)

    def record_total_work(self, value: float):
        self.total_work_duration.observe(value)

    def record_inference(self, value: float):
        self.inference_duration.observe(value)

    def record_warmup_time(self, value: float):
        self.warmup_duration.set(value)

    def record_queue_wait_time(self, value: float):
        self.queue_wait_time.inc(value)

    def shutdown(self):
        self.server.shutdown()
        self.t.join()
