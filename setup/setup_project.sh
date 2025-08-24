#!/bin/bash

# Base project path
BASE_DIR="/media/ali/workspace/workspace/python/deep-distribute-cluster"

# Define all directories to be created
DIRS=(
  "$BASE_DIR/app"
  "$BASE_DIR/app/roles"
  "$BASE_DIR/app/ps"
  "$BASE_DIR/app/worker"
  "$BASE_DIR/app/api"
  "$BASE_DIR/app/comms"
  "$BASE_DIR/app/shared"
  "$BASE_DIR/app/db"
  "$BASE_DIR/ops/docker"
  "$BASE_DIR/ops/k8s"
  "$BASE_DIR/ops/scripts"
  "$BASE_DIR/tests"
)

# Define all files to be created
FILES=(
  "$BASE_DIR/app/__init__.py"
  "$BASE_DIR/app/entrypoint.py"
  "$BASE_DIR/app/roles/parameter_server.py"
  "$BASE_DIR/app/roles/worker.py"
  "$BASE_DIR/app/ps/manager.py"
  "$BASE_DIR/app/ps/aggregator.py"
  "$BASE_DIR/app/ps/state.py"
  "$BASE_DIR/app/ps/scheduler.py"
  "$BASE_DIR/app/ps/streamer.py"
  "$BASE_DIR/app/ps/db_reporter.py"
  "$BASE_DIR/app/worker/trainer.py"
  "$BASE_DIR/app/worker/communicator.py"
  "$BASE_DIR/app/worker/dataloader.py"
  "$BASE_DIR/app/worker/model_builder.py"
  "$BASE_DIR/app/worker/throttle.py"
  "$BASE_DIR/app/api/http.py"
  "$BASE_DIR/app/api/schemas.py"
  "$BASE_DIR/app/comms/websocket_server.py"
  "$BASE_DIR/app/comms/websocket_client.py"
  "$BASE_DIR/app/comms/wire.py"
  "$BASE_DIR/app/comms/codec.py"
  "$BASE_DIR/app/shared/config.py"
  "$BASE_DIR/app/shared/logging.py"
  "$BASE_DIR/app/shared/timers.py"
  "$BASE_DIR/app/shared/utils.py"
  "$BASE_DIR/app/shared/constants.py"
  "$BASE_DIR/app/db/pg.py"
  "$BASE_DIR/app/db/dao_training_job.py"
  "$BASE_DIR/app/keras_catalog_service.py"
  "$BASE_DIR/ops/docker/Dockerfile.ps"
  "$BASE_DIR/ops/docker/Dockerfile.worker"
  "$BASE_DIR/ops/docker/docker-compose.yml"
  "$BASE_DIR/ops/k8s/ps-deployment.yaml"
  "$BASE_DIR/ops/k8s/worker-daemonset.yaml"
  "$BASE_DIR/ops/scripts/run_ps.sh"
  "$BASE_DIR/ops/scripts/run_worker.sh"
  "$BASE_DIR/tests/test_wire_protocol.py"
  "$BASE_DIR/tests/test_aggregator.py"
  "$BASE_DIR/tests/test_ssp_staleness.py"
  "$BASE_DIR/tests/test_db_reporting.py"
  "$BASE_DIR/.env.example"
  "$BASE_DIR/requirements.txt"
  "$BASE_DIR/README.md"
)

# Create directories
for dir in "${DIRS[@]}"; do
  mkdir -p "$dir"
done

# Create empty files
for file in "${FILES[@]}"; do
  touch "$file"
done

echo "✅ Project structure 'deep-distribute-cluster/' created successfully."
