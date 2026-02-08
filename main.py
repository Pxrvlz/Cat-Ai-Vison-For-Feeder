"""
Cat Ai Vison For Feeder
-----------------------
This program uses a camera to find cats and identify them.
If the cat is close enough, it simulates opening a feeder.
"""

# Ketabkhane ha
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import time

# Tanzimat
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

ASSETS_DIR = "assets"
CAT_A_DIR = os.path.join(ASSETS_DIR, "cat_a")
CAT_B_DIR = os.path.join(ASSETS_DIR, "cat_b")

CONFIDENCE_THRESHOLD = 0.5
CLASS_ID_CAT = 15 

# Mohasebe fasele
# Fasele = Zarib FOCAL / Ertefa box
FOCAL_FACTOR = 600.0  
DISTANCE_THRESHOLD = 1.0 

# Tanzimat shenasayi
SIMILARITY_THRESHOLD = 0.8 

# Rang ha
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

def load_yolo_model():
    # Load kardan model YOLO
    print("Loading YOLO...")
    model = YOLO("yolov8n.pt") 
    return model

def load_feature_extractor():
    # Load kardan ResNet baraye shenasayi
    print("Loading ResNet18...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # Faghat feature mikhaim na class
    model.fc = torch.nn.Identity()
    
    model.to(DEVICE)
    model.eval() 
    return model, weights.transforms()

def get_embedding(model, preprocess, image_crop):
    # Tabdil aks be adad (vector)
    if image_crop.size == 0:
        return None

    img_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    try:
        input_tensor = preprocess(img_pil)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = model(input_batch)
            
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_similarity(emb1, emb_list):
    # Moghayese vector ha
    # Bishtarin shabahat ro bar migardone
    if emb1 is None or not emb_list:
        return 0.0
        
    max_sim = 0.0
    
    for emb2 in emb_list:
        if emb2 is None: 
            continue
            
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        
        if norm_a == 0 or norm_b == 0:
            sim = 0.0
        else:
            sim = dot_product / (norm_a * norm_b)
            
        if sim > max_sim:
            max_sim = sim
            
    return max_sim

def estimate_distance(bbox_height):
    # Takhmin fasele ba estefade az ertefa box
    if bbox_height <= 0:
        return 999.0
    return FOCAL_FACTOR / bbox_height

def load_reference_embeddings(directory, model, preprocess):
    # Load kardan hame aks ha
    embeddings = []
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        return embeddings
        
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        if img is None:
            continue
            
        emb = get_embedding(model, preprocess, img)
        if emb is not None:
            embeddings.append(emb)
            
    return embeddings

def main():
    print("Starting Cat Ai Vison For Feeder...")
    
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    
    # Load kardan model ha
    yolo_model = load_yolo_model()
    feature_model, preprocess = load_feature_extractor()
    
    # Load kardan gorbe haye zakhire shode
    emb_list_a = load_reference_embeddings(CAT_A_DIR, feature_model, preprocess)
    emb_list_b = load_reference_embeddings(CAT_B_DIR, feature_model, preprocess)
    
    print(f"Loaded {len(emb_list_a)} images for Cat A")
    print(f"Loaded {len(emb_list_b)} images for Cat B")
    
    if not emb_list_a:
        print("Cat A not set. Press 'a' to add an image.")
    if not emb_list_b:
        print("Cat B not set. Press 'b' to add an image.")

    # Ruhan kardan durbin
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nSystem Ready.")
    print("Press 'q' to quit.")
    print("Press 'a' or 'b' to save cat.")
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # Peyda kardan gorbe ha
        results = yolo_model(frame, stream=True, verbose=False, classes=[CLASS_ID_CAT])
        
        detected_cats = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                    
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cat_crop = frame[y1:y2, x1:x2]
                
                embedding = get_embedding(feature_model, preprocess, cat_crop)
                
                # Shenasayi gorbe
                cat_name = "Unknown"
                similarity_a = calculate_similarity(embedding, emb_list_a)
                similarity_b = calculate_similarity(embedding, emb_list_b)
                
                best_sim = 0.0
                if similarity_a > SIMILARITY_THRESHOLD and similarity_a > similarity_b:
                    cat_name = "Cat A"
                    best_sim = similarity_a
                elif similarity_b > SIMILARITY_THRESHOLD and similarity_b > similarity_a:
                    cat_name = "Cat B"
                    best_sim = similarity_b
                    
                # Mohasebe fasele
                bbox_h = y2 - y1
                distance = estimate_distance(bbox_h)
                
                # Mantegh dar feeder
                lid_status = "CLOSED"
                status_color = COLOR_RED
                
                if distance < DISTANCE_THRESHOLD:
                    if cat_name == "Cat A":
                        lid_status = "OPEN (Lid A)"
                        status_color = COLOR_GREEN
                    elif cat_name == "Cat B":
                        lid_status = "OPEN (Lid B)"
                        status_color = COLOR_GREEN
                    else:
                        lid_status = "WAITING"
                        status_color = COLOR_YELLOW
                
                detected_cats.append({
                    "bbox": (x1, y1, x2, y2),
                    "name": cat_name,
                    "distance": distance,
                    "status": lid_status,
                    "color": status_color,
                    "crop": cat_crop,
                    "embedding": embedding
                })

        # Namayesh ruye safe
        for cat in detected_cats:
            x1, y1, x2, y2 = cat["bbox"]
            color = cat["color"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cat['name']} | {cat['distance']:.2f}m | {cat['status']}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w, y1), color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 2)

            # Navar fasele
            bar_length = int(min(1.0, 1.0 / (cat['distance'] + 0.1)) * 100)
            cv2.rectangle(frame, (x1, y2 + 10), (x1 + 100, y2 + 20), COLOR_BLACK, -1)
            cv2.rectangle(frame, (x1, y2 + 10), (x1 + bar_length, y2 + 20), color, -1)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"Device: {DEVICE} | FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        
        cv2.imshow("Cat Ai Vison For Feeder", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            if detected_cats:
                print("Adding image for Cat A...")
                cat = detected_cats[0]
                
                # Zakhire aks
                timestamp = int(time.time())
                filename = f"cat_a_{timestamp}.jpg"
                path = os.path.join(CAT_A_DIR, filename)
                cv2.imwrite(path, cat["crop"])
                
                # Ezafe kardan be hafeze
                emb_list_a.append(cat["embedding"])
                print(f"Saved {filename}")
                
        elif key == ord('b'):
            if detected_cats:
                print("Adding image for Cat B...")
                cat = detected_cats[0]
                
                # Zakhire aks
                timestamp = int(time.time())
                filename = f"cat_b_{timestamp}.jpg"
                path = os.path.join(CAT_B_DIR, filename)
                cv2.imwrite(path, cat["crop"])
                
                # Ezafe kardan be hafeze
                emb_list_b.append(cat["embedding"])
                print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
