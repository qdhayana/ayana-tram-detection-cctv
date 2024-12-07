from ultralytics import YOLO
import cv2

def detect_tram(
    model_path,
    image_path,
    conf_threshold
):
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Perform detection
    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False
    )
    
    # Draw results
    annotated_frame = results[0].plot()
    
    # Save result
    cv2.imwrite('detected.jpg', annotated_frame)

if __name__ == "__main__":
    model_path = input('Please input the model name or YYYYMMDD_HHMMSS: ')
    model_path = f'runs/train/{model_path}/weights/best.pt'
    image_path = input('Please input the image path: ')
    conf_threshold = float(input('Please input the confidence threshold (0.25 ~ 1.0): '))
    
    detect_tram(model_path, image_path, conf_threshold)