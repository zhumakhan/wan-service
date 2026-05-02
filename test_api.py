import base64
import json
import os

import requests

API_URL = "http://localhost:8001/generate"
OUTPUT_DIR = "test_outputs"


def send_test_request(batch, height, width):
    payload = {
        "job_set_id": "1223023024",
        "s3_region": "eu-north-1",
        "result_queue_url": "link_to_url",
        "sqs_region": "eu-north-1",
        "result_bucket_name": "result_bucket_name",
        "generations": [
            {
                "job_id": "123456543",
                "result_object_key": f"test-user-id/zhuma-test-{b}.png",
                "result_compressed_object_key": f"test-user-id/zhuma-test-{b}_min.webp",
            }
            for b in range(batch)
        ],
        "generation_params": {
            "n_prompt": "journal, text, white borders, lowres, bad quality, intense emotions, overdramatic expression, grimacing, creepy, insane expression, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, three legs, plastic face, artificial-looking face, doll-like face, overly smooth skin, synthetic look, artificial skin, oversaturated skin, unrealistic skin texture, plastic skin",
            "width": width,
            "height": height,
            "sample_shift": 2,
            "sample_solver": "unipc",
            "sample_steps": 20,
            "sample_guide_scale": 3.0,
            "base_seed": 1888,
            "batch_size": batch,
            "use_sage_attention": False,
            "use_apg": True,
            "apg_eta": 0,
            "compile": False,
            "prompt": "man in the red hat",
        },
    }

    print(f"Sending request: batch={batch}, height={height}, width={width}")
    response = requests.post(API_URL, json=payload, timeout=600)

    print(f"Status: {response.status_code}")
    print(f"{batch} x {width} x {height}")

    body = response.json()
    images = body.pop("images", [])
    print(f"Latency: {body['latency']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, b64 in enumerate(images):
        out_path = os.path.join(
            OUTPUT_DIR, f"b{batch}_{width}x{height}_{i}.png"
        )
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"Saved {out_path}")

    return response


if __name__ == "__main__":
    from itertools import product

    shapes = [
        # [1152, 2048],
        # [1536, 2048],
        # [2048, 1536],
        [2048, 1152],
        # [2048, 2048]
    ]
    batch_sizes = [1]
    configs = list(product(batch_sizes, shapes))
    for batch, (w, h) in configs:
        send_test_request(batch, h, w)
# 1xh100
# b  x  width x  height   latency(s)
# 1  x  1152  x  2048     24.08520445181057
# 1  x  1536  x  2048     33.79216889711097
# 1  x  2048  x  2048     48.576420784927905
# 2  x  1152  x  2048     48.01689176494256
# 2  x  1536  x  2048     67.23605569405481
# 2  x  2048  x  2048     97.41096994886175
# 3  x  1152  x  2048     72.17914418084547
# 3  x  1536  x  2048     99.99626521160826
# 3  x  2048  x  2048     143.9316055700183
# 4  x  1152  x  2048     95.02980306372046
# 4  x  1536  x  2048     133.2143324590288
# 4  x  2048  x  2048     193.21078103780746