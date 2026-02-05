import os
import numpy as np
import torch
import torch.nn as nn
import logging
from dataset import pil_loader
from utils import *
from conf import get_config,set_logger,set_env
import cv2
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
def main(conf):
    dataset_info = hybrid_prediction_infolist

    # data
    vid_path = conf.input

    if conf.stage == 1:
        from model.ANFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    else:
        from model.MEFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)
    
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    net.eval()
    # Get participant name from last part of vid_path
    participant_name = os.path.basename(os.path.normpath(vid_path))
    logging.info(f"Processing participant: {participant_name}")
    # Check if directory called participant_name exists in conf.outdir, if not create it
    participant_outdir = os.path.join(conf.outdir, participant_name)  
    if not os.path.exists(participant_outdir):
        os.makedirs(participant_outdir)  
    # Collect all files in vid_path ending with ee.MP4
    for file in os.listdir(vid_path):
        if file.endswith('.mp4'):#'ee.MP4'):
            vid_file_path = os.path.join(vid_path, file)
            # Check if output directory exists, if not create it
            if not os.path.exists(conf.outdir):
                os.makedirs(conf.outdir)
            # Check if an output file already exists
            csv_name = os.path.splitext(file)[0] + '.csv'
            csv_path = os.path.join(conf.outdir, csv_name)
            if os.path.exists(csv_path):
                logging.info(f"Output file {csv_path} already exists. Skipping {vid_file_path}.")
                continue
            # Read .MP4 file from vid_file_path and save every conf.frames_divide frames as images in a temporary folder
            logging.info(f"Processing video file: {vid_file_path}")
            temp_frame_dir = "temp_frames"
            if not os.path.exists(temp_frame_dir):
                os.makedirs(temp_frame_dir)
            vidcap = cv2.VideoCapture(vid_file_path)
            success, image = vidcap.read() 
            count = 0
            saved_count = 0 
            while success:
                if count % conf.frames_divide == 0:
                    frame_path = os.path.join(temp_frame_dir, f"frame{saved_count:05d}.jpg")
                    cv2.imwrite(frame_path, image)     # save frame as JPEG file
                    saved_count += 1
                success, image = vidcap.read()                
                count += 1
            vidcap.release()
            logging.info(f"Extracted {count} frames from video.")

            # Process each frame and save results in dataframe
            results = []
            for i in tqdm(range(saved_count)):
                frame_path = os.path.join(temp_frame_dir, f"frame{i:05d}.jpg")
                img_transform = image_eval()
                img = pil_loader(frame_path)
                img_ = img_transform(img).unsqueeze(0)

                if torch.cuda.is_available():
                    net = net.cuda()
                    img_ = img_.cuda()

                with torch.no_grad():
                    pred = net(img_)
                    pred = pred.squeeze().cpu().numpy()

                results.append(pred)
                #logging.info(f"Processed frame {i+1}/{count}")
            

            # Clean up temporary frame directory
            for i in range(saved_count):
                frame_path = os.path.join(temp_frame_dir, f"frame{i:05d}.jpg")
                os.remove(frame_path)   
            os.rmdir(temp_frame_dir)
            logging.info("Cleaned up temporary frames.")

            # Average predictions over all frames
            pred = np.mean(np.array(results), axis=0)

            # log
            infostr = {'AU prediction:'}
            logging.info(infostr)
            infostr_probs,  infostr_aus = dataset_info(pred, 0.5)
            logging.info(infostr_aus)
            logging.info(infostr_probs)

            AU_list = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11', 'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20',
                    'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39', 
                    'AUL1', 'AUR1', 'AUL2', 'AUR2', 'AUL4', 'AUR4', 'AUL6', 'AUR6', 'AUL10', 
                    'AUR10', 'AUL12', 'AUR12', 'AUL14', 'AUR14']
            
            # Save prediction as csv
            # Create CSV path based on file name
            csv_name = os.path.splitext(file)[0] + '.csv'
            csv_path = os.path.join(conf.outdir, participant_name, csv_name)


            #csv_path = conf.input.split('.')[0] + '_predictions.csv'

            # Convert results to DataFrame with shape (len(AU_list, count)) and save as CSV
            # Each column corresponds to an AU, each row to a frame 
            df = pd.DataFrame(np.array(results), columns=AU_list)
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved predictions to {csv_path}")
    


# ---------------------------------------------------------------------------------

if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

