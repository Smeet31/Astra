# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:59:26 2022

@author: smith
"""

# import the necessary packages
from collections import deque
import numpy as np
import imutils
import cv2
import torch 
import boto3
import torch.nn as nn
import os
import time
import torchvision

class gpu():
    

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(model, device):
        """Move tensor(s) to chosen device"""
        if isinstance(model, (list,tuple)):
            return [gpu.to_device(x, device) for x in model]
        return model.to(device, non_blocking=True)
    
    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield gpu.to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)


# IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)

SAMPLE_DURATION = 16
SAMPLE_SIZE = 224
# initialize the frames queue used to store a rolling sample duration
# of frames -- this queue will automatically pop out old frames and
# accept new ones
# load the human activity recognition model
model_state = torch.load(r'C:\Users\smith\Documents\BE_SMITH\Code\V4_ALL\all\har_resnet.pt')
s3 = boto3.resource('s3')

class_labels= ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 14)
print(model)
model.load_state_dict(model_state)
print('past model loaded')

device=gpu.get_default_device()
print(device)
model=gpu.to_device(model, device)

frames = deque(maxlen=SAMPLE_DURATION)

def lambda_handler(event, context):
  bucket_name = event['Records'][0]['s3']['bucket']['name']
  key = event['Records'][0]['s3']['object']['key']

  video = readVideoFromBucket(key, bucket_name).resize(SAMPLE_SIZE)

  vs = cv2.VideoCapture(video)

  # used to record the time at which we processed current frame
  while torch.no_grad():
      model.eval()
      (grabbed, frame) = vs.read()
      if not grabbed:
          print('No frame read--exit()')
          break
      frame = imutils.resize(frame, width=800)

      frames.append(frame)
      new_frame_time = time.time()
      if len(frames) < SAMPLE_DURATION:
          continue
      blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE))

      # print(blob.shape)

      blob = np.transpose(blob, (1, 0, 2, 3))
      # print(blob.shape)

      blob = torch.from_numpy(blob)

      blob = blob.to(device)

      blob = blob.unsqueeze(0)
      blob = model(blob)
      pred = torch.max(blob, dim=1)[1].tolist()
      preds = torch.max(blob, dim=1)[1]
      label = class_labels[pred[0]]
      probs = torch.softmax(blob, dim=1)
      prob = probs[0][preds.item()]

      # converting the fps to string so that we can display it on frame
      # by using putText function

      if prob.item() > 0.75:
          a = str(round((prob.item() * 100), 2))
          b = " "
          label = label + b + a + b

          print(f'Model Prediction {label}')

      elif prob.item() < 0.75:
          a = str(round((prob.item() * 100), 2))
          b = " "
          f = 'Not Sure='
          label = f + label + b + a + b

          print(f'No sure  {label}')

  vs.release()

  # Closes all the frames
  cv2.destroyAllWindows()


def readVideoFromBucket(key, bucket_name):
  bucket = s3.Bucket(bucket_name)
  object = bucket.Object(key)
  response = object.get()
  return Image.open(response['Body'])

