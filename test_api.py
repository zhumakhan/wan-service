import json
import requests

API_URL = "http://localhost:8001/generate"


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
            "sample_guide_scale": 1.0,
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
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response


if __name__ == "__main__":
    from itertools import product

    shapes = [
        # [1152, 2048],
        # [1536, 2048],
        [2048, 2048]
    ]
    batch_sizes = [4]
    configs = list(product(batch_sizes, shapes))
    for batch, (h, w) in configs:
        send_test_request(batch, h, w)
