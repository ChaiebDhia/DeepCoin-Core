import sys, os
with open('ENGINEERING_JOURNAL.md', 'rb') as f:
    content_bytes = f.read()

text_to_insert = b'''
EXPLAINABILITY & ACTIVE LEARNING FILES:
src/core/gradcam.py                 Implements Grad-CAM++ to generate heatmap visuals showing
                                    which exact pixels the CNN focused on. Exists to build trust
                                    by making the "black box" model explainable to historians.

scripts/active_learning.py          Extracts unconfident and edge-case predictions from production
                                    logs to build a continuous retraining loop. Exists so the
                                    model learns from real-world failures automatically.

DEVOPS & INFRASTRUCTURE FILES:
docker-compose.yml                  The 7-Service orchestration blueprint. Connects FastAPI,
                                    Next.js, PostgreSQL, Redis, MLflow, LocalStack, and Nginx.
                                    Exists to mirror the cloud production environment locally.

Dockerfile.api                      Python 3.12-slim backend container. Strips out unused
                                    OS dependencies and locks the PyTorch/FastAPI runtime.
                                    Protects against directory traversals and environment drift.

frontend/Dockerfile                 Node 22-alpine frontend container. Leverages Next.js 
                                    'standalone' output tracing to compress a 600MB app into 
                                    a 150MB secure image, resolving all npm CVE vulnerabilities.

.github/workflows/ci.yml            Continuous Integration pipeline. Automatically runs over
                                    120 Pytest assertions and Linters on every GitHub Push.
                                    Exists to block broken code from the main pipeline.

.github/workflows/cd.yml            Continuous Deployment pipeline. Automatically triggers
                                    strict Trivy vulnerability scans before pushing secure
                                    Docker images to the GitHub Container Registry (GHCR).
'''

target_string = b'.github/copilot-instructions.md'

if target_string in content_bytes:
    idx = content_bytes.find(target_string) + len(target_string)
    idx = content_bytes.find(b'milestone.', idx) + 10
    
    new_content = content_bytes[:idx] + b'\n' + text_to_insert + content_bytes[idx:]
    with open('ENGINEERING_JOURNAL.md', 'wb') as f:
        f.write(new_content)
    print('Inserted successfully')
else:
    print('Target not found')
