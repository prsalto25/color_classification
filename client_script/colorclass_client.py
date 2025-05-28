import zmq
import cv2
import numpy as np
import time

class ColorClassificationClient:
    def __init__(self, url="tcp://localhost:8890", timeout=5000):  
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)  
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)  
        self.socket.connect(url)
        self.connected = True  
        print(f"Connected to {url}")  

    def detect_colors(self, frame):
        if not self.connected:  
            return np.array([[]], dtype=np.float32)
            
        try:  
            if frame is None or frame.size == 0:
                return np.array([[]], dtype=np.float32)
            
            h, w, c = frame.shape
            frame_bytes = frame.tobytes()
            shape_bytes = f"{h}_{w}_{c}".encode()
            
            self.socket.send_multipart([frame_bytes, shape_bytes])
            
            response = self.socket.recv_pyobj()
            return response
            
        except zmq.error.Again:  
            print("Timeout waiting for server response")
            return np.array([[]], dtype=np.float32)
        except Exception as e:  
            print(f"Error: {e}")
            return np.array([[]], dtype=np.float32)
    
    def is_connected(self):  
        """Check if client is connected"""
        return self.connected
    
    def close(self):  
        """Close connection"""
        if self.connected:
            self.socket.close()
            self.context.term()
            self.connected = False
            print("Connection closed")

if __name__ == "__main__":
    client = ColorClassificationClient()
    try:  
        frame = cv2.imread("test1.jpg")
        if frame is not None:
            results = client.detect_colors(frame)
            print(f"Detections: {len(results)} objects")
        else:
            print("Please provide test1.jpg for testing")
            
    finally:  
        client.close()