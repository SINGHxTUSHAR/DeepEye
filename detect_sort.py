# import necessary libraries 
import argparse # for parsing command line arguements 
import time # for time related operations 
from pathlib import Path # for handling file path 

import cv2  # OpenCV library for computer vision tasks
import torch  # PyTorch deep learning framework
import torch.backends.cudnn as cudnn  # Backend for CuDNN (CUDA Deep Neural Network library)

# Setting environment variables
import os  # For interacting with the operating system
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # Allowing duplicate libraries for OpenMP

import sys  # For system-specific parameters and functions
sys.path.insert(0, './yolov5') # Setting path to YOLOv5 module without changing base

from yolov5.models.experimental import attempt_load # loading yolo v5 model 
from yolov5.utils.datasets import LoadStreams, LoadImages # loading image/video data for yolo v5 model 
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box # general utilities for yolo v5 model 
from yolov5.utils.plots import colors, plot_one_box # utilities to plot bounding boxes in yolo v5 
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized # custom function to visualize bounding boxes


from deep_sort_pytorch.utils.parser import get_config  # Parsing configurations for DeepSort
from deep_sort_pytorch.deep_sort import DeepSort  # DeepSort object tracker

from graphs import bbox_rel, draw_boxes  # Custom functions for visualizing bounding boxes

# Function for detecting objects using YOLOv5 and applying DeepSort for tracking
@torch.no_grad() # Decorator for disabling gradient calculation in PyTorch
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='yolov5/data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml"  #Deep Sort configuration
           ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Check if the source is a webcam, a file with a .txt extension, or a URL/stream
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize DeepSort by loading configurations from a specified file
    cfg = get_config()  # Load default DeepSort configurations
    cfg.merge_from_file(opt.config_deepsort)  # Update configurations from the specified file
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,  # Path to the ReID (Re-identification) model checkpoint
        max_dist=cfg.DEEPSORT.MAX_DIST,  # Maximum allowable distance between object embeddings for matching
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,  # Minimum confidence score for object detections
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,  # Maximum overlap allowed for Non-Maximum Suppression (NMS)
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,  # Maximum IoU (Intersection over Union) distance
        max_age=cfg.DEEPSORT.MAX_AGE,  # Maximum age of an object's track
        n_init=cfg.DEEPSORT.N_INIT,  # Number of consecutive frames to activate the tracker
        nn_budget=cfg.DEEPSORT.NN_BUDGET,  # Size of the appearance descriptor distance metric cache
        use_cuda=True  # Specify to use CUDA (GPU) for processing if available
    )
    
    
        # Directories
    # Increment the path for saving results, creating a new directory if needed
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)

    # Create 'labels' directory within 'save_dir' if save_txt is True, otherwise create 'save_dir'
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    # Set up logging
    set_logging()

    # Select the appropriate device (CPU or CUDA GPU)
    device = select_device(device)

    # Check if half precision is supported and applicable for the current device
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # Attempt to load the FP32 model from specified weights, using the selected device
    model = attempt_load(weights, map_location=device)

    # Determine the stride of the model for scaling purposes
    stride = int(model.stride.max())  # model stride

    # Check and adjust the image size if necessary based on the model's stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Get class names from the model
    # Checking if the model is a multi-GPU model and accessing class names accordingly
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Convert the model to FP16 (half-precision) if applicable
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # This section initializes and loads a second-stage classifier (currently set to False)
    classify = False

    # If the classifier is enabled, load and initialize the classifier model
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


    # Set Dataloader
    vid_path, vid_writer = None, None

    # Check if the source is a webcam or other media file (image or video)
    if webcam:
        # Check if the user wants to view images during processing
        view_img = check_imshow()
        
        # Speed up constant image size inference if using CUDA
        cudnn.benchmark = True  # set True to speed up constant image size inference
        
        # Load a video stream if the source is a webcam
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        # Load images or a video file as the dataset
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference once on an empty tensor to initialize the model
    # This step is useful for CUDA-based devices to set up the model's parameters
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Record the start time of the inference process
    t0 = time.time()

    
    
        ##for frame_idx enumerate
        # Iterate through frames and relevant data in the dataset
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        # Convert the image to a PyTorch tensor and move it to the selected device (GPU or CPU)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # Convert uint8 to fp16/32
        img /= 255.0  # Normalize pixel values from 0 - 255 to 0.0 - 1.0

        # Check if the image has 3 dimensions, if not, add an extra dimension
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Time synchronization for performance measurement
        t1 = time_synchronized()

        # Perform inference on the model using the processed image
        pred = model(img, augment=augment)[0]

        # Apply Non-Maximum Suppression to filter detections based on thresholds and other parameters
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Record time after Non-Maximum Suppression
        t2 = time_synchronized()

        # Apply classification if enabled
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
# Process detections
    for i, det in enumerate(pred):  # Iterate through detections per image
        if webcam:  # Condition for batch_size >= 1
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # Convert 'p' to a Path object
        save_path = str(save_dir / p.name)  # Create path to save image (img.jpg)
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # Create path to save label (img.txt)
        s += '%gx%g ' % img.shape[2:]  # Prepare string for printing image dimensions
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
        imc = im0.copy() if save_crop else im0  # Create a copy of the image for saving cropped predictions
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print detection results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # Count detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Add class count to the string

            bbox_xywh = []
            confs = []
            # Adapt detections to deep sort input format
            for *xyxy, conf, cls in det:
                x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)  # Extract coordinates and dimensions
                obj = [x_c, y_c, bbox_w, bbox_h]  # Create bounding box object [x_center, y_center, width, height]
                bbox_xywh.append(obj)  # Append bounding box object to list
                confs.append([conf.item()])  # Append confidence score to list


                
                xywhs = torch.Tensor(bbox_xywh)  # Convert bounding box data to a PyTorch tensor
                confss = torch.Tensor(confs)  # Convert confidence scores to a PyTorch tensor

                # Pass detections to deepsort for object tracking
                outputs = deepsort.update(xywhs, confss, im0)

                # Draw boxes for visualization if there are any detected objects
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]  # Extract the bounding box coordinates
                    identities = outputs[:, -1]  # Extract the identities of the objects
                    draw_boxes(im0, bbox_xyxy, identities)  # Draw bounding boxes on the image

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]  # Left coordinate of the bounding box
                        bbox_top = output[1]  # Top coordinate of the bounding box
                        bbox_w = output[2]  # Width of the bounding box
                        bbox_h = output[3]  # Height of the bounding box
                        identity = output[-1]  # Identity of the object
                        with open(txt_path, 'a') as f:
                            # Write the object's information in the MOT label format to the file
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                
                
                # Write results Label
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Check if saving to a file is required
                        # Convert bounding box coordinates to normalized xywh format
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        # Define the label format based on whether confidence scores need to be saved
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:  # Open file to write label information
                            # Write the label information to the file in the specified format
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Check if saving images or crops or viewing images is required
                        c = int(cls)  # Get the integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # Define the label to display on the image
                        # Plot the bounding box on the image with the specified label and color
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:  # Check if saving cropped images is required
                            # Save the cropped bounding box as an image
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            
            else:
                deepsort.increment_ages() # increment ages for deepsort tracking 
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

           # Stream results
            if view_img:  # Check if displaying images is required
                cv2.imshow(str(p), im0)  # Show the image with detections
                cv2.waitKey(1)  # Wait for a specified amount of time to display the image

            # Save results (image with detections)
            if save_img: # check if saving images is required
                if dataset.mode == 'image': # check if dataset is an image
                    cv2.imwrite(save_path, im0) # save image with detections
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path   # # Set the new video path
                        if isinstance(vid_writer, cv2.VideoWriter):  # If the video writer exists
                            vid_writer.release()  # Release the previous video writer
                        if vid_cap:  # If it's a video
                            # Get video properties: fps, width, and height
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # If it's a stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # Set default properties for stream
                            save_path += '.mp4'  # Add .mp4 extension for the stream file'
                            #  Initialize a video writer to save the video with detections
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0) # Write the frame with detections to the video file

        # Check if saving text files or images is required
        if save_txt or save_img:
            # Conditionally generate a string to display the number of labels saved
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")  # Print a message indicating where the results are saved

        # If 'update' flag is set, strip optimizer from the weights file to fix SourceChangeWarning
        if update:
            strip_optimizer(weights)  # Update the model by stripping the optimizer

        # Print total execution time
        print(f'Done. ({time.time() - t0:.3f}s)')  # Display the total execution time of the script



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='ped.mp4', help='file/dir/URL/glob, 0 for webcam')

    # Inference Configuration
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.60, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')

    # Device Configuration
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # Output and Display Configuration
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')

    # Filtering and Class Configuration
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # Model Updating and Naming Configuration
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    # Visualization Configuration
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')

    # Performance Configuration
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")

    # Parse the command line arguments
    opt = parser.parse_args()

    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))
