#!/bin/bash
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

portIn=8890     
portOut=8891   

echo "Starting ZMQ Color Detection System..."
echo "Client port: $portIn"
echo "Server port: $portOut"

echo "Starting broker..."
python3 broker.py $portIn $portOut & 
BROKER_PID=$!

sleep 3  

echo "Starting YOLO+Color server..."
python3 yolo_ultralytics_zmq.py &
SERVER_PID=$!

echo "System ready! Press Ctrl+C to stop."
cleanup() {
    echo "Shutting down system..."
    kill $BROKER_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    wait
    echo "System stopped."
}
trap cleanup EXIT INT TERM
wait