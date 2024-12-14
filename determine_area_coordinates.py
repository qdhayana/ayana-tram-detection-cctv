import cv2
import numpy as np
import signal
import sys

class AreaSelector:
    def __init__(self, _points):
        self._points = _points
        self.points = []
        self.img = None
        self.img_copy = None
        self.window_name = f'Define Area A - Click {_points} points'
        
    def cleanup(self):
        """Cleanup all opencv windows"""
        cv2.destroyAllWindows()
        # Sometimes destroyAllWindows() doesn't work right away
        for i in range(self._points+1):
            cv2.waitKey(1)
    
    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.points.append([x, y])
            # Draw point
            cv2.circle(self.img, (x, y), 5, (0, 255, 0), -1)
            
            # Draw lines between points
            if len(self.points) > 1:
                cv2.line(self.img, tuple(self.points[-2]), tuple(self.points[-1]), (0, 255, 0), 2)
            
            # If we have X points, connect the last point to the first
            if len(self.points) == self._points:
                cv2.line(self.img, tuple(self.points[-1]), tuple(self.points[0]), (0, 255, 0), 2)
            
            # Update image display
            cv2.imshow(self.window_name, self.img)
    
    def define_area_coordinates(self, image_path):
        try:
            # Read the image
            self.img = cv2.imread(image_path)
            if self.img is None:
                print(f"Error: Could not read image from {image_path}")
                return np.array([])
                
            self.img_copy = self.img.copy()  # Keep a clean copy
            
            # Create window and set mouse callback
            cv2.imshow(self.window_name, self.img)
            cv2.setMouseCallback(self.window_name, self.click_event)
            
            print(f"Click {self._points} points to define Area A. Press 'r' to reset, 'c' to confirm, 'q' to quit")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                # Reset points
                if key == ord('r'):
                    self.points = []
                    self.img = self.img_copy.copy()
                    cv2.imshow(self.window_name, self.img)
                
                # Confirm selection
                elif key == ord('c') and len(self.points) == self._points:
                    break
                
                # Quit
                elif key == ord('q'):
                    self.points = []
                    break
                
            return np.array(self.points)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return np.array([])
            
        finally:
            self.cleanup()

def save_area_coordinates(points, _points, filename='area_coordinates.txt'):
    """Save coordinates to a file"""
    if len(points) == _points:
        np.savetxt(filename, points, fmt='%d')
        print(f"Coordinates saved to {filename}")
    else:
        print("No coordinates to save")

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    print('\nInterrupted! Cleaning up...')
    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)
    sys.exit(0)

def main(_points):
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        image_path = 'ayana_tram_stop_get_coordinates.jpg'
        
        # Create selector instance
        selector = AreaSelector(_points)
        
        # Define area interactively
        points = selector.define_area_coordinates(image_path)
        
        if len(points) == _points:
            # Save coordinates
            save_area_coordinates(points, _points)
            print("Area A coordinates:", points.tolist())
        else:
            print("Area definition cancelled or incomplete")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        # Ensure windows are cleaned up
        cv2.destroyAllWindows()
        for i in range(_points):
            cv2.waitKey(1)

if __name__ == "__main__":
    main(_points=8)