# Separate verifier image. Owns ALL verifier dependencies, pinned. The agent
# environment (/app) is mounted at run time and is never copied or mutated here.
FROM python:3.11-slim

# Pinned verifier dependencies. No runtime installs happen in test.sh.
RUN pip install --no-cache-dir \
        pytest==8.3.3 \
        jsonschema==4.23.0 \
        numpy==2.2.6

# Verifier code + hidden fixtures live only inside this image.
WORKDIR /tests
COPY . /tests

# Deterministic, non-interactive test entry point.
CMD ["bash", "/tests/test.sh"]
