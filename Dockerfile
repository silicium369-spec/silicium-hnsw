ARG BASE_IMAGE=python:3.11-slim
FROM --platform=linux/amd64 ${BASE_IMAGE}

ARG GITHUB_TOKEN=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates file util-linux g++ && \
    mkdir -p /usr/local/lib && \
    if [ -n "$GITHUB_TOKEN" ]; then \
        curl -fsSL -H "Authorization: token $GITHUB_TOKEN" \
            -o /usr/local/lib/libsilicium_ann_x86_64.so \
            "https://github.com/silicium369-spec/silicium-hnsw/releases/download/ann-v1.0.0/libsilicium_ann_x86_64.so"; \
    else \
        curl -fsSL -o /usr/local/lib/libsilicium_ann_x86_64.so \
            "https://github.com/silicium369-spec/silicium-hnsw/releases/download/ann-v1.0.0/libsilicium_ann_x86_64.so"; \
    fi && \
    echo "11bfdd30bfa78fbffc2b9d6ae88b3ea18310a274d684a167e410ca17b266ee58  /usr/local/lib/libsilicium_ann_x86_64.so" | sha256sum -c - && \
    ldconfig && \
    pip install --no-cache-dir -q numpy pyyaml usearch hnswlib faiss-cpu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/
RUN if [ -d /home/app ]; then \
        mkdir -p /home/app/ann_benchmarks/algorithms/silicium && \
        cp -f /workspace/module.py /workspace/config.yml /home/app/ann_benchmarks/algorithms/silicium/; \
    fi

RUN python3 -c 'import ctypes; ctypes.CDLL("/usr/local/lib/libsilicium_ann_x86_64.so"); print("[silicium] Build environment verified")'

CMD ["./reproduce_all.sh"]
