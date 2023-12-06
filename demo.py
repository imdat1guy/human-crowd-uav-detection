import argparse
import numpy as np
import torch
import os
import cv2
import h5py
from PIL import Image
from src.crowd_net import CrowdNet
from torchvision import transforms

def main(path, weights):
    # Load the model
    model = CrowdNet()
    model = model.cuda()
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])

    # Directory containing frames
    maps_dir = os.path.join(path, 'density_map/')
    frame_dir = os.path.join(path, 'images')
    print(frame_dir)
    frame_files = sorted(os.listdir(frame_dir))
    print(len(frame_files))
    
    output_path = 'outputs/output_video.avi'
    # Initialize video writer
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    frame_height, frame_width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (frame_width*2, frame_height))  # Adjust FPS as needed

    i = 0
    total = len(frame_files)

    for file in frame_files:
        i += 1
        print(f'Processing Frame [{i}/{total}]')
        frame = Image.open(os.path.join(frame_dir,file)).convert('RGB')

        # Preprocess frame
        transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ])
        input_tensor = transform(frame).cuda().unsqueeze(0)

        # Inference
        with torch.no_grad():
            density_map = model(input_tensor)
            density_map = density_map.detach().cpu()
            count = int(density_map.sum().numpy())

        # Resize density map to match frame size
        density_map_np = np.asarray(density_map.reshape(density_map.shape[2],density_map.shape[3]))
        density_map_np = np.clip(density_map_np / np.max(density_map_np), 0, 1)  # Normalize
        # density_map_np = (density_map_np * 255).astype(np.uint8)  # Brighten


        # Apply color map for visualization
        density_map_colored = cv2.applyColorMap(np.uint8(255 * density_map_np), cv2.COLORMAP_JET)
        density_map_resized = cv2.resize(density_map_colored, (frame_width, frame_height))

        # Overlay on frame
        overlayed_frame = cv2.addWeighted(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR), 0.5, density_map_resized, 0.5, 0)
        cv2.putText(overlayed_frame, f'Count: {int(count)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Process the original frame for display
        original_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # GT
        ll=os.path.splitext(os.path.basename(file))[0]
        temp = h5py.File(maps_dir + ll + '.h5', 'r')
        temp_1 = np.asarray(temp['density'])
        gt_count = int(np.sum(temp_1)) + 1
        cv2.putText(original_frame, f'GT Count: {int(gt_count)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Concatenate frames
        combined_frame = np.hstack((original_frame, overlayed_frame))

        # Write frame to video
        out.write(combined_frame)

    # Release the video writer
    out.release()
    print(f"Completed Processing Demo. Output can be found in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description.")
    parser.add_argument("--path", type=str, default="Data/VisDrone2020-prepared/val", help="Path to the val data split")
    parser.add_argument("--weights", type=str, default="pretrained/task_one_model_best.pth.tar", help="Path to the weights file.")

    args = parser.parse_args()
    main(args.path, args.weights)
