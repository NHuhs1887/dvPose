"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author:
    Gaurvi Goyal

LICENSE GOES HERE
"""
import argparse
import cv2
import os
import numpy as np
import sys

import torch

import dv_processing as dv
import time
import datetime
from datetime import timedelta

sys.path.append('.')
sys.path.append('./hpe-core')
#
# from lib import init, MoveNet, Task
from datasets.utils.events_representation import EROS, eventFrame
# from datasets.h36m.utils.parsing import movenet_to_hpecore
from datasets.utils.export import str2bool

from pycore.moveenet import init, MoveNet, Task

from pycore.moveenet.config import cfg
from pycore.moveenet.visualization.visualization import add_skeleton



run_task = None
model = None
filter_chain = None
visualizer = None
accumulator = None
rep = None



def get_representation(rep_name, args):
    if rep_name == 'eros':
        rep = EROS(kernel_size=args.eros_kernel, frame_width=args.frame_width, frame_height=args.frame_height)
    elif rep_name == 'ef':
        rep = eventFrame(frame_height=args.frame_height, frame_width=args.frame_width, n=args.n)
    else:
        print('Representation not found for this setup.')
        exit()
    return rep

def convertevents(eventStore):
    newevents = dict()
    newevents['ts'] = eventStore.timestamps() / 1e6
    coordinates = eventStore.coordinates()
    newevents['x'] = coordinates[:,0]
    newevents['y'] = eventStore.coordinates()[:,1]
    newevents['pol'] = eventStore.polarities()

    return newevents


def process(eventstore):
    filter_chain.accept(eventstore)
    eventstore = filter_chain.generateEvents()
    events = dict()
    starttime = time.time()
    events = convertevents(eventstore)
    
    rep.reset_frame()
    print('Time taken to convert: ', time.time()-starttime)
    
    starttime = time.time()
    for ei in range(eventstore.size() - 1):
        rep.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))
    print('Time taken to update: ', time.time()-starttime)
    frame = rep.get_frame()


    frame = cv2.GaussianBlur(frame, (args.gauss_kernel, args.gauss_kernel), 0)
    starttime = time.time()
    pre = run_task.predict_online(frame)
    print('Time taken to predict: ', time.time()-starttime)
    output = np.concatenate((pre['joints'].reshape([-1,2]), pre['confidence'].reshape([-1,1])), axis=1)

    frame = add_skeleton(frame, output, (0, 0, 255), True, normalised=False)

    cv2.imshow('', frame)
    cv2.waitKey(1)

    return

# def accumulateAndProcess(eventstore):
#     accumulator.accept(events)
#     frame = accumulator.generateFrame()

    
#     #frame = cv2.GaussianBlur(frame, (args.gauss_kernel, args.gauss_kernel), 0)
#     starttime = time.time()
#     pre = run_task.predict_online(frame)
#     print('Time taken to predict: ', time.time()-starttime)
#     output = np.concatenate((pre['joints'].reshape([-1,2]), pre['confidence'].reshape([-1,1])), axis=1)

#     frame = add_skeleton(frame, output, (0, 0, 255), True, normalised=False)

#     cv2.imshow('', frame)
#     cv2.waitKey(1)


def preview(eventstore):
    filter_chain.accept(eventstore)
    eventstore = filter_chain.generateEvents()
    frame = visualizer.generateImage(events)
    cv2.imshow("Preview", frame)
    cv2.waitKey(1)



    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-eros_kernel', help='EROS kernel size', default=8, type=int)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-gauss_kernel', help='Gaussian filter for EROS', default=7, type=int)
    parser.add_argument('-input', help='Path to input folder (with the data.log file in it)', default=None, type=str)
    parser.add_argument("-write_video", type=str, default=None, help="Set path with file name to save video")
    parser.add_argument("-ckpt", type=str, default='models/e97_valacc0.81209.pth', help="path to the ckpt. Default: MoveEnet checkpoint.")
    parser.add_argument('-fps', help='Output frame rate', default=50, type=int)
    parser.add_argument('-stop', help='Set to an integer value to stop early after these frames', default=None, type=int)
    parser.add_argument('-rep', help='Representation eros or ef', default='eros', type=str)
    parser.add_argument('-n', help='Number of events in constant count event frame [7500]', default=7500, type=int)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=False, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=1.0, type=float)
    parser.add_argument('-visualise', type=str2bool, nargs='?', default=True, help="Visualise Results [TRUE]")

    
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
        exit()
    #reader = dv.io.MonoCameraRecording(args.input)


    if not torch.cuda.is_available():
        print("No cuda available")
    





    cfg['ckpt'] = args.ckpt
    if args.dev:
        args.input = '/home/ggoyal/data/h36m_cropped/tester/h5/cam2_S1_Directions_1/Directions.h5'
        args.write_video = '/home/ggoyal/data/tester.mp4'
        args.stop = 100

    if args.input == None:
        print('Starting live feed')
        capture = dv.io.CameraCapture()
    else:
        capture = dv.io.MonoCameraRecording(args.input)
        input_data_dir = os.path.abspath(args.input)
        print("=====", input_data_dir, "=====")
        if not os.path.exists(input_data_dir):
            print(input_data_dir, 'does not exist')
        
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=50), process)

    filter_chain = dv.EventFilterChain()
        # Filter refractory period
    filter_chain.addFilter(dv.RefractoryPeriodFilter(capture.getEventResolution(), timedelta(milliseconds=1)))
        # Only positive events
    filter_chain.addFilter(dv.EventPolarityFilter(True))
        # Remove noise
    filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(capture.getEventResolution(), backgroundActivityDuration=timedelta(milliseconds=1)))

    visualizer = dv.visualization.EventVisualizer(capture.getEventResolution())

    accumulator = dv.Accumulator(capture.getEventResolution())

    # Apply configuration, these values can be modified to taste
    accumulator.setMinPotential(0.0)
    accumulator.setMaxPotential(1.0)
    accumulator.setNeutralPotential(0.5)
    accumulator.setEventContribution(0.15)
    accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
    accumulator.setDecayParam(1e+6)
    accumulator.setIgnorePolarity(False)
    accumulator.setSynchronousDecay(False)

    rep = get_representation(args.rep, args)


    init(cfg)
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    run_task = Task(cfg, model)
    run_task.modelLoad(cfg['ckpt'])



    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
       #slicer

