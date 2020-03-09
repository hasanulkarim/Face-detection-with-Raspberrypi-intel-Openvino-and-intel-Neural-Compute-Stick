import argparse
import cv2
import numpy as np
from inference import Network
import time

INPUT_STREAM = "/home/pi/Documents/NCS_playground/test_video.mp4"
model_path = "/home/pi/Documents/NCS_playground/models/face_detection/face-detection-adas-0001.xml"
##For WINDOWS
#CPU_EXTENSION = "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Debug\cpu_extension_avx2.dll"
##FOr linux:
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
##For MYRIAD:
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/armv7l/libmyriadPlugin.so"
#For raspian Armv71:
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/armv7l/libcpu_extension.so"
#For raspian Armv71 hetero?:
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/armv7l/libHeteroPlugin.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-m", help=m_desc, default=model_path)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='MYRIAD')
    args = parser.parse_args()

    return args


def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    plugin = Network()
    
   
    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()
    
    #check if input is a webcam and set args.i=0, so that cv2.videocapture() can use the system camera
    if args.i == 'CAM':
        args.i = 0
    

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux 0x00000021
    #For linux:
    out = cv2.VideoWriter('cam_out_5fps.mp4', 0x00000021, 5, (width,height)) #this is for linux, for mac use ('M','J','P','G')
    ## For windows:
    #fourcc = cv2.VideoWriter_fourcc(*'MPG4')
    #out = cv2.VideoWriter('out.mp4',fourcc, 30, (width,height))
    
    
    start_time = time.time()
    framecount = 0
    duration = 10
    # Process frames until the video ends, or process is exited
    while (int(time.time() - start_time) < duration):
        # Read the next frame
        flag, frame = cap.read()
        
        #count the number of frames
        framecount += 1
        
        if not flag:
            break
        key_pressed = cv2.waitKey(10)
        ### TODO: Pre-process the frame
#         # I am using the semantic segmentation-adas-0001 model, so preprocessing will be done according to that model
#         n_w = 2048
#         n_h = 1024
#         preprocessed = np.copy(frame)
#         print('=========')
#         print('frame.shape=',preprocessed.shape)
#         print('net_input_shape=',net_input_shape)
#         print('=========')
#         preprocessed = cv2.resize(preprocessed,n_w,n_h)
#         preprocessed = preprocessed.transpose((2,0,1))
#         preprocessed = preprocessed.reshape(1,3,n_h,n_w)

        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        ### TODO: Perform inference on the frame
        
        plugin.async_inference(p_frame)
        

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()

        ### TODO: Update the frame to include detected bounding boxes
        frame = draw_boxes(frame, result, args, width, height)
        # Write out the frame
        out.write(frame)
        
        # Break if escape key pressed
        if key_pressed == 27:
            break
    print('Total frames: {} in {} seconds'.format(framecount, duration))
    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def draw_boxes(frame, result, args, width, height):
        '''
        Draw bounding boxes onto the frame.
        '''
        for box in result[0][0]: # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= 0.5:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
        return frame
    
def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
