# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:19:30 2019

@author: foram.jivani
"""

import json
import cv2

import pandas as pd

import numpy as np
#from blink_detection_final import detect
import time
import pathlib
import os
import zipfile
import threading
import logging
    # mainfolder_path : /a/b/c/test/
    # video_pathIn :  /a/b/c/test/test.MOV
    # faceinfo_jsonFile : /a/b/c/test/test.json
    # video_fileName : test
    # sourceFile : /d/e/f/PAN.jpg


def generate(mainfolder_path,video_pathIn,faceinfo_jsonFile,video_fileName,sourceFile,jsonString):
    
    folder_pathIn = mainfolder_path 
    #in_json = 'test.json'
    loginfo = ''
    loginfo += "##############################" +'\n'
    loginfo += 'main folder where data is stored = '+ str(folder_pathIn) +'\n'
    loginfo += "##############################" + '\n' 
    
    
    pose_csvPath = folder_pathIn + video_fileName + '_pose.csv'
    smile_csvPath = folder_pathIn + video_fileName + '_smile.csv'
    blink_csvPath = folder_pathIn + video_fileName + '_blink.csv'
    
    path_leftFaces = folder_pathIn + 'faceImages/left face poses/'
    pathlib.Path(path_leftFaces).mkdir(parents=True, exist_ok=True)    

    path_rightFaces = folder_pathIn + 'faceImages/right face poses/'
    pathlib.Path(path_rightFaces).mkdir(parents=True, exist_ok=True)
    
    path_upFaces = folder_pathIn + 'faceImages/up face poses/'
    pathlib.Path(path_upFaces).mkdir(parents=True, exist_ok=True)
    
    path_downFaces = folder_pathIn + 'faceImages/down face poses/'
    pathlib.Path(path_downFaces).mkdir(parents=True, exist_ok=True)
    
    path_tiltleftFaces = folder_pathIn + 'faceImages/tilt left face poses/'
    pathlib.Path(path_tiltleftFaces).mkdir(parents=True, exist_ok=True)
    
    path_tiltrightFaces = folder_pathIn + 'faceImages/tilt right face poses/'
    pathlib.Path(path_tiltrightFaces).mkdir(parents=True, exist_ok=True)
    
    path_frontFaces = folder_pathIn + 'faceImages/front face poses/'
    pathlib.Path(path_frontFaces).mkdir(parents=True, exist_ok=True)
    
    #path_smilingFaces = folder_pathIn + 'smilingfaces/'
    #pathlib.Path(path_smilingFaces).mkdir(parents=True, exist_ok=True)
    
    #path_blinkingFaces = folder_pathIn + 'blinkingfaces/'
    #pathlib.Path(path_blinkingFaces).mkdir(parents=True, exist_ok=True)
    
    path_allFaces = folder_pathIn + 'faceImages/allframes/'
    pathlib.Path(path_allFaces).mkdir(parents=True, exist_ok=True)
    
    path_targetFaces = folder_pathIn + 'faceImages/target/'
    pathlib.Path(path_targetFaces).mkdir(parents=True, exist_ok=True)
    
    output_json =folder_pathIn + 'output_'+ video_fileName +'.json'
    
    st1 = time.time()
    
    front_starttime = -1
    front_endtime = -1
    
    left_starttime = -1
    left_endtime = -1
        
    right_starttime = -1
    right_endtime = -1        

    up_starttime = -1
    up_endtime = -1
        
    down_starttime = -1
    down_endtime = -1        
    
    tiltleft_starttime = -1
    tiltleft_endtime = -1        

    tiltright_starttime = -1
    tiltright_endtime = -1        

    smile_starttime = -1
    smile_endtime = -1
        
    blink_starttime = -1
    blink_endtime = -1
    
    d = json.loads(json.dumps(jsonString))
    
    sorted_d = sorted(map(lambda x: (x[0], int(x[1])), d.items()), key=lambda x: x[1])
    swapped_d = map(lambda x: (x[1], x[0]), sorted_d)
    timed_d = {}
    
    for k, v in swapped_d:
        if k in timed_d:
            timed_d[k].append(v)
        else:
            timed_d[k] = [v]
    
    times = sorted(list(timed_d.keys()))
    
    
    tmp = {}
    
    for idx, tm in enumerate(times):
        for side in timed_d[tm]:
            tmp[side] = (tm, times[idx+1] if idx+1 != len(times) else -1)        

    for key, (val,nxtVal) in tmp.items():
        value = val
    
        nextValue = nxtVal
        
        if key == 'front':
            front_starttime = value
            front_endtime = nextValue
        
        if key == 'left':
            left_starttime = value
            left_endtime = nextValue
            
        if key == 'right':
            right_starttime = value
            right_endtime = nextValue
            
        if key == 'up':
            up_starttime = value
            up_endtime = nextValue
            
        if key == 'down':
            down_starttime = value
            down_endtime = nextValue
            
        if key == 'tilt left':
            tiltleft_starttime = value
            tiltleft_endtime = nextValue
            
        if key == 'tilt right':
            tiltright_starttime = value
            tiltright_endtime = nextValue
            
        if key == 'smile':
            smile_starttime = value
            smile_endtime = nextValue
            
        if key == 'blink':
            blink_starttime = value
            blink_endtime = nextValue    
    
    #pathOut = 'E:/Study/Mtech study/capturedImages2_size3'
    
    #noofclust = 3
    
    loginfo += "##############################" +'\n'
    loginfo += "time taken to process json string ====="+ str(time.time()-st1)+ '\n'
    loginfo += "##############################" + '\n'

    st1 = time.time()
    
    with open(faceinfo_jsonFile) as f:
        data = json.load(f)
    

    faceFrames = []
    
    X = [[]]
    datafr = [[]]
    smileinfo=[[]]
    eyesopeninfo = [[]]
    
    data_ts_front_pose =[[]]
    
    data_ts_left_pose =[[]]
    data_ts_right_pose =[[]]
    
    data_ts_up_pose =[[]]
    data_ts_down_pose =[[]]
    
    data_ts_tiltleft_pose =[[]]
    data_ts_tiltright_pose =[[]]
    
    cnt_closeeyes = 0
    cnt_openeyes = 0
    
    flag_eyeware = False
    eyeware_val = 'no'
    datadict = {}
    
    ############         calculate sqrt distance to each axis  #################
    for x in data['Faces']:
        #if 'Face' in x['Person']:
            faceFrames.append(x['Timestamp'])
            #readable = (x['Timestamp']/1000)%60
            #pprint(readable)
            timestamp = x['Timestamp']
            pitch = x['Face']['Pose']['Pitch']
            roll = x['Face']['Pose']['Roll']
            yaw = x['Face']['Pose']['Yaw']
            
            smile_val = str(x['Face']['Smile']['Value'])
            #print(smile_val)
            smile_confidence = x['Face']['Smile']['Confidence']
            if smile_starttime <= timestamp and (timestamp < smile_endtime or smile_endtime == -1 )and (smile_starttime != -1):
                if smile_val == 'True':
                    smileinfo.append([x['Timestamp'], smile_val,smile_confidence])
            
            eyesopen_val = str(x['Face']['EyesOpen']['Value'])
            eyesopen_confidence = x['Face']['EyesOpen']['Confidence']
            
            if blink_starttime <= timestamp and (timestamp < blink_endtime or blink_endtime == -1) and (blink_starttime != -1):
                
                if flag_eyeware == False :
                    eyeglasses_val = str(x['Face']['Eyeglasses']['Value'])
                    sunglasses_val = str(x['Face']['Sunglasses']['Value'])
                    
                    if eyeglasses_val == 'True' or sunglasses_val == 'True' :
                        eyeware_val = 'yes'
                    else:
                        eyeware_val = 'no'
                    flag_eyeware = True
                
                if eyesopen_val == 'False' :
                    eyesopeninfo.append([x['Timestamp'], eyesopen_val,eyesopen_confidence])
                    cnt_closeeyes = cnt_closeeyes +1
                else:
                    cnt_openeyes = cnt_openeyes + 1
                
            X.append([pitch ,roll ,yaw])
            
            datafr.append([x['Timestamp'],pitch ,roll ,yaw ])
            
            if front_starttime <= timestamp and (timestamp < front_endtime or front_endtime == -1) and (front_starttime != -1 ):
                distance = np.sqrt((pitch)**2+(roll)**2+ (yaw)**2)
                data_ts_front_pose.append([x['Timestamp'], distance])
            
            if up_starttime <= timestamp and (timestamp < up_endtime or up_endtime == -1) and (up_starttime != -1 ):            
                distance = np.sqrt((180-pitch)**2+(roll)**2+ (yaw)**2)
                data_ts_up_pose.append([x['Timestamp'], distance])
            
            if down_starttime <= timestamp and (timestamp < down_endtime or down_endtime ==-1) and (down_starttime != -1 ):
                distance = np.sqrt((180+pitch)**2+(roll)**2+ (yaw)**2)
                data_ts_down_pose.append([x['Timestamp'], distance])
            
            if left_starttime <= timestamp and (timestamp < left_endtime or left_endtime == -1) and (left_starttime != -1 ):
                distance = np.sqrt((pitch)**2+(roll)**2+ (180-yaw)**2)
                data_ts_left_pose.append([x['Timestamp'], distance])
            
            if right_starttime <= timestamp and (timestamp < right_endtime or right_endtime == -1) and (right_starttime != -1 ):
                distance = np.sqrt((pitch)**2+(roll)**2+ (180+yaw)**2)
                data_ts_right_pose.append([x['Timestamp'], distance])
            
            if tiltleft_starttime <= timestamp and (timestamp < tiltleft_endtime or tiltleft_endtime == -1)  and (tiltleft_starttime != -1 ):
                distance = np.sqrt((pitch)**2+(180-roll)**2+ (yaw)**2)
                data_ts_tiltleft_pose.append([x['Timestamp'], distance])
            
            if tiltright_starttime <= timestamp and (timestamp < tiltright_endtime or tiltright_endtime == -1) and (tiltright_starttime != -1 ):
                distance = np.sqrt((pitch)**2+(180+roll)**2+ (yaw)**2)
                data_ts_tiltright_pose.append([x['Timestamp'], distance])
            
            datadict[x['Timestamp']] = { 
            'AgeRange' : x['Face']['AgeRange'],        
            'Smile' : x['Face']['Smile'],
            'Eyeglasses' : x['Face']['Eyeglasses'],
            'Sunglasses' : x['Face']['Sunglasses'],
            'Gender' : x['Face']['Gender'],
            'Beard' : x['Face']['Beard'],
            'Mustache' : x['Face']['Mustache'],
            'EyesOpen' : x['Face']['EyesOpen'],
            'MouthOpen' : x['Face']['MouthOpen'],
            'Emotions' : x['Face']['Emotions'],
            'Quality' : x['Face']['Quality']
            }
    
    noofface=len(faceFrames)    # total no of face having pitch roll yaw info in video 
    loginfo += '####################   No of faces  ==   '+ str(noofface) +'\n'
        
    #print(len(nonface))     # total no of non face 
    
    datafr.remove([])
    smileinfo.remove([])
    eyesopeninfo.remove([])
    # format[ timestamp , distance ]
    data_ts_front_pose.remove([])   
    data_ts_up_pose.remove([])
    data_ts_down_pose.remove([])
    
    data_ts_left_pose.remove([])
    data_ts_right_pose.remove([])
    
    data_ts_tiltleft_pose.remove([])
    data_ts_tiltright_pose.remove([])
    
    loginfo += '############################' +'\n'
    loginfo += 'no of frames where eyesopen = false ::::::: '+ str(cnt_closeeyes)+'\n'
    loginfo += '############################'+ '\n'
    loginfo += 'no of frames where eyesopen = true ::::::: ' + str(cnt_openeyes) + '\n'
    loginfo += '############################' + '\n'
    
    # save data of all faces in csv file
    df = pd.DataFrame(datafr,columns=['Timestamp','Pitch','Roll','Yaw'])
    if(any(df.duplicated('Timestamp'))):
        loginfo += 'Multiple person detected.......' + '\n'
        data_out = {}
        data_out = 'Multiple person detected.......'
    else:    
        df.to_csv(pose_csvPath, index=False)
        
        #to get the maximum angle between face pose and axis sort in ascending minimum distance 
        data_ts_front_pose = sorted(data_ts_front_pose, key=lambda x: x[1])
        data_ts_up_pose = sorted(data_ts_up_pose, key=lambda x: x[1])
        data_ts_down_pose = sorted(data_ts_down_pose, key=lambda x: x[1])
        
        data_ts_left_pose = sorted(data_ts_left_pose, key=lambda x: x[1])
        data_ts_right_pose = sorted(data_ts_right_pose, key=lambda x: x[1])
        
        data_ts_tiltleft_pose = sorted(data_ts_tiltleft_pose, key=lambda x: x[1])
        data_ts_tiltright_pose = sorted(data_ts_tiltright_pose, key=lambda x: x[1])
    
      
        front_cnt = 0
        left_cnt = 0
        right_cnt = 0
        up_cnt = 0
        down_cnt = 0
        tiltleft_cnt = 0
        tiltright_cnt = 0
        
        front_flg = 'no'
        left_flg = 'no'
        right_flg = 'no'
        up_flg = 'no'
        down_flg = 'no'
        tiltleft_flg = 'no'
        tiltright_flg = 'no'
        smile_flg = 'no'
        blink_flg = 'no'
        
        ##############     Pose Thresholds     ########################
        left_dist_thre = 145
        right_dist_thre = 145
        up_dist_thre = 145
        down_dist_thre = 145
        tiltleft_dist_thre = 145
        tiltright_dist_thre = 145
        front_dist_thre = 15
        down_dist_thre = 145
    
    
        smile_thre = 95
        if len(smileinfo) != 0:
            smileinfo = sorted(smileinfo, key=lambda x: x[2],reverse=True)
            df_smileinfo = pd.DataFrame(smileinfo,columns=['Timestamp','Smile','Confidence'])
            df_smileinfo.to_csv(smile_csvPath, index=False)
            if (smileinfo[0][2] >= smile_thre):
                ts_smile_face = smileinfo[0][0]
                smile_flg='yes'
        else:
            smile_flg = 'no'
        
        blink_thre = 85
        if len(eyesopeninfo) != 0:
            eyesopeninfo = sorted(eyesopeninfo, key=lambda x: x[2],reverse=True)
            df_eyesopeninfo = pd.DataFrame(eyesopeninfo,columns=['Timestamp','Blink','Confidence'])
            df_eyesopeninfo.to_csv(blink_csvPath, index=False)
            if (eyesopeninfo[0][2] >= blink_thre):
                ts_blink_face = eyesopeninfo[0][0]
                blink_flg='yes'
            else:
                blink_flg = 'no'
        
        '''
        #get timestamp for minimum distance value 
        ts_front_face = data_ts_front_pose[0][0]
        ts_up_face = data_ts_up_pose[0][0]
        ts_down_face = data_ts_down_pose[0][0]
        
        ts_left_face = data_ts_left_pose[0][0]
        ts_right_face = data_ts_right_pose[0][0]
        
        ts_tiltleft_face = data_ts_tiltleft_pose[0][0]
        ts_tiltright_face = data_ts_tiltright_pose[0][0]
        '''
        
        loginfo += "##############################" +'\n'
        loginfo += "time taken to process json data  and pose csv nd smile csv ====="+  str(time.time()-st1)+ '\n'
        loginfo += "##############################" + '\n' 
        
        
        st1 = time.time()
        vidcap = cv2.VideoCapture(video_pathIn)
        success,image = vidcap.read()
        success = True
        
        
        
        ###########   left face poses images    ##############################   
        pathOut = path_leftFaces
        datafr_len = len(data_ts_left_pose)
        if datafr_len != 0:
            dis = data_ts_left_pose[0][1]
            ts = data_ts_left_pose[0][0]
            i=0
            while dis <= left_dist_thre:
                #print(ts)
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
                left_cnt = left_cnt + 1
                i=i+1
                if i < datafr_len :
                    dis = data_ts_left_pose[i][1]
                    ts = data_ts_left_pose[i][0]
                else:
                    break
            
            if (i!=0):
                ts_left_face = data_ts_left_pose[i-1][0]
                #print('ts_left_face =',ts_left_face)
                left_flg = 'yes'
        
        ###########   right face poses images    ##############################   
        pathOut = path_rightFaces
        datafr_len = len(data_ts_right_pose)
        if datafr_len != 0:
            dis = data_ts_right_pose[0][1]
            ts = data_ts_right_pose[0][0]
            i=0
            while dis <= right_dist_thre:
                #print(ts)
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
                right_cnt = right_cnt + 1
                i = i+1
                if i < datafr_len :        
                    dis = data_ts_right_pose[i][1]
                    ts = data_ts_right_pose[i][0]
                else:
                    break
            
            if (i!=0):
                ts_right_face = data_ts_right_pose[i-1][0]
                #print('ts_right_face =',ts_right_face)
                right_flg = 'yes'
        
        ###########   up face poses images    ##############################   
        pathOut = path_upFaces
        datafr_len = len(data_ts_up_pose)
        if datafr_len != 0:
            dis = data_ts_up_pose[0][1]
            ts = data_ts_up_pose[0][0]
            i=0
            while dis <= up_dist_thre:
                #print(ts)
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
                up_cnt = up_cnt + 1
                i = i+1
                if i < datafr_len :
                    dis = data_ts_up_pose[i][1]
                    ts = data_ts_up_pose[i][0]
                else:
                    break
            
            if (i!=0):
                ts_up_face = data_ts_up_pose[i-1][0]
                #print('ts_up_face =',ts_up_face)
                up_flg= 'yes'
        
        ###########   down face poses images    ##############################   
        pathOut = path_downFaces
        datafr_len = len(data_ts_down_pose)
        if datafr_len != 0:
            dis = data_ts_down_pose[0][1]
            ts = data_ts_down_pose[0][0]
            i=0
            while dis <= down_dist_thre:
                #print(ts)
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
                down_cnt = down_cnt + 1
                i = i+1
                if i < datafr_len :
                    dis = data_ts_down_pose[i][1]
                    ts = data_ts_down_pose[i][0]
                else:
                    break
            
            if(i!=0):
                ts_down_face = data_ts_down_pose[i-1][0]
                #print('ts_down_face =',ts_down_face)
                down_flg = 'yes'
        
        ###########   tilt left face poses images    ##############################   
        pathOut = path_tiltleftFaces
        datafr_len = len(data_ts_tiltleft_pose)
        if datafr_len != 0:
            dis = data_ts_tiltleft_pose[0][1]
            ts = data_ts_tiltleft_pose[0][0]
            i=0
            while dis <= tiltleft_dist_thre:
                #print(ts)
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
                tiltleft_cnt = tiltleft_cnt + 1
                i = i+1    
                if i < datafr_len :
                    dis = data_ts_tiltleft_pose[i][1]
                    ts = data_ts_tiltleft_pose[i][0]
                else:
                    break
            
            if(i!=0):
                ts_tiltleft_face = data_ts_tiltleft_pose[i-1][0]
                #print('ts_tiltleft_face =',ts_tiltleft_face)
                tiltleft_flg = 'yes'
        
        ###########   tilt right face poses images    ##############################   
        pathOut = path_tiltrightFaces
        datafr_len = len(data_ts_tiltright_pose)
        if datafr_len != 0:
            dis = data_ts_tiltright_pose[0][1]
            ts = data_ts_tiltright_pose[0][0]
            i=0
            while dis <= tiltright_dist_thre:
                #print(ts)
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
                tiltright_cnt = tiltright_cnt + 1
                i = i+1
                if i < datafr_len :
                    dis = data_ts_tiltright_pose[i][1]
                    ts = data_ts_tiltright_pose[i][0]
                else:
                    break
            
            if(i!=0):
                ts_tiltright_face = data_ts_tiltright_pose[i-1][0]
                #print('ts_tiltright_face =',ts_tiltright_face)
                tiltright_flg = 'yes'
        
        ###########   front face poses images    ##############################   
        pathOut = path_frontFaces
        datafr_len = len(data_ts_front_pose)
        if datafr_len != 0:
            dis = data_ts_front_pose[0][1]
            ts = data_ts_front_pose[0][0]
            i=0
            while dis <= front_dist_thre:
                #print(ts)
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
                front_cnt = front_cnt + 1
                i = i+1
                if i < datafr_len :
                    dis = data_ts_front_pose[i][1]
                    ts = data_ts_front_pose[i][0]
                else:
                    break
            
            if(i!=0):
                ts_front_face = data_ts_front_pose[i-1][0]
                #print('ts_front_face =',ts_front_face)
                front_flg = 'yes'
            
         
        
        ##################              TARGET FACES            ###################################
        pathOut = path_targetFaces
        
        if(front_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_front_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "front_" +"%d.jpg" % ts_front_face, image) 
        
        if(up_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_up_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "up_" + "%d.jpg" % ts_up_face, image) 
        
        if(down_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_down_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "down_" + "%d.jpg" % ts_down_face, image) 
        
        if(left_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_left_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "left_" + "%d.jpg" % ts_left_face, image) 
        
        if(right_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_right_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "right_" + "%d.jpg" % ts_right_face, image) 
        
        if(tiltleft_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_tiltleft_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "tiltleft_" + "%d.jpg" % ts_tiltleft_face, image) 
        
        if(tiltright_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_tiltright_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "tiltright_" + "%d.jpg" % ts_tiltright_face, image) 
        
        
        if(smile_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_smile_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "smile_" + "%d.jpg" % ts_smile_face, image) 
        
        if(blink_flg == 'yes'):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts_blink_face))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "blink_" + "%d.jpg" % ts_blink_face, image) 
        
            
            
        ########### blink detection    ##############################   
    
        
        #start1 = time.time()    
        #blink_flg = detect(mainfolder_path,video_pathIn,path_targetFaces)
        #print("time taken by blink detection:", (time.time()-start1))
         
        
        
        ###########  saving all face images    ##############################   
        
        pathOut = path_allFaces
        datafr_len = len(datafr)
        for i in range(datafr_len):
            ts = datafr[i][0]
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
            success,image = vidcap.read()
            #print ('Read a new frame: ', success)
            cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
        
        
         
        '''
        ###########    Saving all SMILE FACES    #################################
        if(smile_flg == 'yes'):    
            pathOut = path_smilingFaces
            datafr_len = len(smileinfo)
            for i in range(datafr_len):
                ts = smileinfo[i][0]
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
            
          
        '''  
        
        '''
        ###########    Saving all Blinked FACES    #################################
        if(blink_flg == 'yes'):    
            pathOut = path_blinkingFaces
            datafr_len = len(eyesopeninfo)
            for i in range(datafr_len):
                ts = eyesopeninfo[i][0]
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(ts))    # added this line 
                success,image = vidcap.read()
                print ('Read a new frame: ', success)
                cv2.imwrite( pathOut + "%d.jpg" % ts, image) 
            
          
        '''  
        
        loginfo += "##############################" +'\n'
        loginfo += "time taken to save all pose images ===== "+ str(time.time()-st1) + '\n'
        loginfo += "##############################" + '\n'
        
        
        '''
        
        print('front : ', front_flg)
        print('left : ', left_flg)
        print('right : ', right_flg)
        print('up : ', up_flg)
        print('down : ', down_flg)
        print('tilt left : ', tiltleft_flg)
        print('tilt right : ', tiltright_flg)
        print('smile : ', smile_flg)
        print('blink : ', blink_flg)
         
        '''
        
        ##############         Comapre faces    #################
        start = time.time()
    
        import boto3
        
        from os import walk
        
        front_match = 'not available'
        left_match = 'not available'
        right_match = 'not available'
        tiltleft_match = 'not available'
        tiltright_match = 'not available'
        up_match = 'not available'
        down_match = 'not available'
        smile_match= 'not available'
        blink_match= 'not available'
        confidence = 0
        
        front_confidence = 0
        left_confidence = 0
        right_confidence = 0
        tiltleft_confidence = 0
        tiltright_confidence = 0
        up_confidence = 0
        down_confidence = 0
        smile_confidence = 0
        blink_confidence = 0
        
        
        f = []
        mypath = path_targetFaces
        for (dirpath, dirnames, filenames) in walk(mypath):
            f.extend(filenames)
            break
        
        if any(".DS_Store" in s for s in f):
            f.remove('.DS_Store')
        #targetFile='E:/Study/Mtech study/capturedImages2/frame19193.jpg'
        
        st2 = time.time()
        
        zip_fname = video_fileName+'.zip'
        with zipfile.ZipFile(folder_pathIn + zip_fname, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in f:
                zipf.write(mypath+file,os.path.basename(mypath+file))
        
    
        s3Resource = boto3.resource('s3')
        def upload():
            try:
                s3Resource.meta.client.upload_file(folder_pathIn+zip_fname, 'kristal-ai-demo-targets', zip_fname)
                logging.info("time taken to upload target in s3 ======  %s", str(time.time()-st2))
            except Exception as err:
                logging.info(err)
        
    
        t = threading.Thread(target = upload).start()
        
         
        client=boto3.client('rekognition')
        
        for i in range(len(f)):
            #print(f[i])
            a =f[i].split('_')
            #sourceFile='/Users/foram.jivani/Desktop/Demo2/id.jpg'
            imageSource=open((folder_pathIn + sourceFile),'rb')
            confidence = 0 
            #targetFile='/Users/foram.jivani/Desktop/Kristal/demo4/faceImages/target/5372.jpg'
            targetFile = mypath + f[i]
            imageTarget=open(targetFile,'rb')
            response=client.compare_faces(SimilarityThreshold=70,SourceImage={'Bytes': imageSource.read()},
                                              TargetImage={'Bytes': imageTarget.read()})
            
            for faceMatch in response['FaceMatches']:
                position = faceMatch['Face']['BoundingBox']
                confidence = str(faceMatch['Face']['Confidence'])
                #print('The face at ' + str(position['Left']) + ' ' + str(position['Top']) +
                #           ' matches with ' + confidence + '% confidence')
            if(float(confidence) >= 75):
                if(a[0] == 'front' and front_flg == 'yes'):
                    front_match = 'yes'
                    front_confidence = (confidence)
                    #print('front',front_confidence)
                if(a[0] == 'left' and left_flg == 'yes'):
                    left_match = 'yes'
                    left_confidence = (confidence)
                    #print('left',left_confidence)
                if(a[0] == 'right' and right_flg == 'yes'):
                    right_match = 'yes'
                    right_confidence = (confidence)
                    #print('right',right_confidence)
                if(a[0] == 'tiltleft' and tiltleft_flg == 'yes'):
                    tiltleft_match = 'yes'
                    tiltleft_confidence = (confidence)
                    #print('tiltleft',tiltleft_confidence)
                if(a[0] == 'tiltright' and tiltright_flg == 'yes'):
                    tiltright_match ='yes'
                    tiltright_confidence = (confidence)
                    #print(tiltright_confidence)
                if(a[0] == 'up' and up_flg == 'yes'):
                    up_match = 'yes'
                    up_confidence = (confidence)
                if(a[0] == 'down' and down_flg == 'yes'):
                    down_match = 'yes'
                    down_confidence = (confidence)
                if(a[0] == 'smile' and smile_flg == 'yes'):
                    smile_match = 'yes'
                    smile_confidence = (confidence)
                if(a[0] == 'blink' and blink_flg == 'yes'):
                    blink_match = 'yes'
                    blink_confidence = (confidence)
            else:
                if(a[0] == 'front' and front_flg == 'yes'):
                    front_match = 'no'
                if(a[0] == 'left' and left_flg == 'yes'):
                    left_match = 'no'
                if(a[0] == 'right' and right_flg == 'yes'):
                    right_match = 'no'
                if(a[0] == 'tiltleft'and tiltleft_flg == 'yes'):
                    tiltleft_match = 'no'
                if(a[0] == 'tiltright' and tiltright_flg == 'yes'):
                    tiltright_match ='no'
                if(a[0] == 'up' and up_flg == 'yes'):
                    up_match = 'no'
                if(a[0] == 'down' and down_flg == 'yes'):
                    down_match = 'no'
                if(a[0] == 'smile' and smile_flg == 'yes'):
                    smile_match = 'no'
                if(a[0] == 'blink' and blink_flg == 'yes'):
                    blink_match = 'no'
        
        
            imageTarget.close() 
         
        #print("##############################") 
        
        data_out = {}
        data_out['front'] ={}
        data_out['front']['detected'] = front_flg
        data_out['front']['match'] = front_match
        data_out['front']['confidence'] = front_confidence 
        
        data_out['left'] ={}
        data_out['left']['detected'] = left_flg
        data_out['left']['match'] = left_match
        data_out['left']['confidence'] = left_confidence 
        
        data_out['right'] ={}
        data_out['right']['detected'] = right_flg
        data_out['right']['match'] = right_match
        data_out['right']['confidence'] = right_confidence 
        
        data_out['tiltleft'] ={}
        data_out['tiltleft']['detected'] = tiltleft_flg
        data_out['tiltleft']['match'] = tiltleft_match
        data_out['tiltleft']['confidence'] = tiltleft_confidence 
        
        data_out['tiltright'] ={}
        data_out['tiltright']['detected'] = tiltright_flg
        data_out['tiltright']['match'] = tiltright_match
        data_out['tiltright']['confidence'] = tiltright_confidence 
        
        data_out['up'] ={}
        data_out['up']['detected'] = up_flg
        data_out['up']['match'] = up_match
        data_out['up']['confidence'] = up_confidence 
        
        data_out['down'] ={}
        data_out['down']['detected'] = down_flg
        data_out['down']['match'] = down_match
        data_out['down']['confidence'] = down_confidence 
        
        data_out['smile'] ={}
        data_out['smile']['detected'] = smile_flg
        data_out['smile']['match'] = smile_match
        data_out['smile']['confidence'] = smile_confidence 
        
        data_out['blink'] ={}
        data_out['blink']['detected'] = blink_flg
        data_out['blink']['match'] = blink_match
        data_out['blink']['confidence'] = blink_confidence 
        data_out['blink']['eyeware'] = eyeware_val
        
        
        if(front_flg == 'yes') :
            face = datadict[ts_front_face]
            data_out['front']['AgeRange'] = {}
            data_out['front']['Smile'] = {}
            data_out['front']['Eyeglasses'] = {}
            data_out['front']['Sunglasses'] = {}
            data_out['front']['Gender'] = {}
            data_out['front']['Beard'] = {}
            data_out['front']['Mustache'] = {}
            data_out['front']['EyesOpen'] = {}
            data_out['front']['MouthOpen'] ={}
            data_out['front']['Emotions'] =[]
                
            data_out['front']['AgeRange'] = face['AgeRange']        
            data_out['front']['Smile'] = face['Smile']
            data_out['front']['Eyeglasses'] = face['Eyeglasses']
            data_out['front']['Sunglasses'] = face['Sunglasses']
            data_out['front']['Gender'] = face['Gender']
            data_out['front']['Beard'] = face['Beard']
            data_out['front']['Mustache'] = face['Mustache']
            data_out['front']['EyesOpen'] = face['EyesOpen']
            data_out['front']['MouthOpen'] = face['MouthOpen']
            data_out['front']['Emotions'] = face['Emotions']
            data_out['front']['Quality'] = face['Quality']
    
        if(up_flg == 'yes') :  
            face = datadict[ts_up_face]
            data_out['up']['AgeRange'] = {}
            data_out['up']['Smile'] = {}
            data_out['up']['Eyeglasses'] = {}
            data_out['up']['Sunglasses'] = {}
            data_out['up']['Gender'] = {}
            data_out['up']['Beard'] = {}
            data_out['up']['Mustache'] = {}
            data_out['up']['EyesOpen'] = {}
            data_out['up']['MouthOpen'] ={}
            data_out['up']['Emotions'] =[]
            
            data_out['up']['AgeRange'] = face['AgeRange']        
            data_out['up']['Smile'] = face['Smile']
            data_out['up']['Eyeglasses'] = face['Eyeglasses']
            data_out['up']['Sunglasses'] = face['Sunglasses']
            data_out['up']['Gender'] = face['Gender']
            data_out['up']['Beard'] = face['Beard']
            data_out['up']['Mustache'] = face['Mustache']
            data_out['up']['EyesOpen'] = face['EyesOpen']
            data_out['up']['MouthOpen'] = face['MouthOpen']
            data_out['up']['Emotions'] = face['Emotions']
            data_out['up']['Quality'] = face['Quality']
        
        if(down_flg == 'yes') :
            face = datadict[ts_down_face]
            data_out['down']['AgeRange'] = {}
            data_out['down']['Smile'] = {}
            data_out['down']['Eyeglasses'] = {}
            data_out['down']['Sunglasses'] = {}
            data_out['down']['Gender'] = {}
            data_out['down']['Beard'] = {}
            data_out['down']['Mustache'] = {}
            data_out['down']['EyesOpen'] = {}
            data_out['down']['MouthOpen'] ={}
            data_out['down']['Emotions'] =[]
            
            data_out['down']['AgeRange'] = face['AgeRange']        
            data_out['down']['Smile'] = face['Smile']
            data_out['down']['Eyeglasses'] = face['Eyeglasses']
            data_out['down']['Sunglasses'] = face['Sunglasses']
            data_out['down']['Gender'] = face['Gender']
            data_out['down']['Beard'] = face['Beard']
            data_out['down']['Mustache'] = face['Mustache']
            data_out['down']['EyesOpen'] = face['EyesOpen']
            data_out['down']['MouthOpen'] = face['MouthOpen']
            data_out['down']['Emotions'] = face['Emotions']
            data_out['down']['Quality'] = face['Quality']
            
        if(left_flg == 'yes') :
            face = datadict[ts_left_face]
            data_out['left']['AgeRange'] = {}
            data_out['left']['Smile'] = {}
            data_out['left']['Eyeglasses'] = {}
            data_out['left']['Sunglasses'] = {}
            data_out['left']['Gender'] = {}
            data_out['left']['Beard'] = {}
            data_out['left']['Mustache'] = {}
            data_out['left']['EyesOpen'] = {}
            data_out['left']['MouthOpen'] ={}
            data_out['left']['Emotions'] =[]
            
            data_out['left']['AgeRange'] = face['AgeRange']        
            data_out['left']['Smile'] = face['Smile']
            data_out['left']['Eyeglasses'] = face['Eyeglasses']
            data_out['left']['Sunglasses'] = face['Sunglasses']
            data_out['left']['Gender'] = face['Gender']
            data_out['left']['Beard'] = face['Beard']
            data_out['left']['Mustache'] = face['Mustache']
            data_out['left']['EyesOpen'] = face['EyesOpen']
            data_out['left']['MouthOpen'] = face['MouthOpen']
            data_out['left']['Emotions'] = face['Emotions']
            data_out['left']['Quality'] = face['Quality']
        
        if(right_flg == 'yes') :  
            face = datadict[ts_right_face]
            data_out['right']['AgeRange'] = {}
            data_out['right']['Smile'] = {}
            data_out['right']['Eyeglasses'] = {}
            data_out['right']['Sunglasses'] = {}
            data_out['right']['Gender'] = {}
            data_out['right']['Beard'] = {}
            data_out['right']['Mustache'] = {}
            data_out['right']['EyesOpen'] = {}
            data_out['right']['MouthOpen'] ={}
            data_out['right']['Emotions'] =[]
            
            data_out['right']['AgeRange'] = face['AgeRange']        
            data_out['right']['Smile'] = face['Smile']
            data_out['right']['Eyeglasses'] = face['Eyeglasses']
            data_out['right']['Sunglasses'] = face['Sunglasses']
            data_out['right']['Gender'] = face['Gender']
            data_out['right']['Beard'] = face['Beard']
            data_out['right']['Mustache'] = face['Mustache']
            data_out['right']['EyesOpen'] = face['EyesOpen']
            data_out['right']['MouthOpen'] = face['MouthOpen']
            data_out['right']['Emotions'] = face['Emotions']
            data_out['right']['Quality'] = face['Quality']
        
        if(tiltleft_flg == 'yes') :   
            face = datadict[ts_tiltleft_face]
            data_out['tiltleft']['AgeRange'] = {}
            data_out['tiltleft']['Smile'] = {}
            data_out['tiltleft']['Eyeglasses'] = {}
            data_out['tiltleft']['Sunglasses'] = {}
            data_out['tiltleft']['Gender'] = {}
            data_out['tiltleft']['Beard'] = {}
            data_out['tiltleft']['Mustache'] = {}
            data_out['tiltleft']['EyesOpen'] = {}
            data_out['tiltleft']['MouthOpen'] ={}
            data_out['tiltleft']['Emotions'] =[]
            
            data_out['tiltleft']['AgeRange'] = face['AgeRange']        
            data_out['tiltleft']['Smile'] = face['Smile']
            data_out['tiltleft']['Eyeglasses'] = face['Eyeglasses']
            data_out['tiltleft']['Sunglasses'] = face['Sunglasses']
            data_out['tiltleft']['Gender'] = face['Gender']
            data_out['tiltleft']['Beard'] = face['Beard']
            data_out['tiltleft']['Mustache'] = face['Mustache']
            data_out['tiltleft']['EyesOpen'] = face['EyesOpen']
            data_out['tiltleft']['MouthOpen'] = face['MouthOpen']
            data_out['tiltleft']['Emotions'] = face['Emotions']
            data_out['tiltleft']['Quality'] = face['Quality']
    
        if(tiltright_flg == 'yes') :
            face = datadict[ts_tiltright_face]
            data_out['tiltright']['AgeRange'] = {}
            data_out['tiltright']['Smile'] = {}
            data_out['tiltright']['Eyeglasses'] = {}
            data_out['tiltright']['Sunglasses'] = {}
            data_out['tiltright']['Gender'] = {}
            data_out['tiltright']['Beard'] = {}
            data_out['tiltright']['Mustache'] = {}
            data_out['tiltright']['EyesOpen'] = {}
            data_out['tiltright']['MouthOpen'] ={}
            data_out['tiltright']['Emotions'] =[]
            
            data_out['tiltright']['AgeRange'] = face['AgeRange']        
            data_out['tiltright']['Smile'] = face['Smile']
            data_out['tiltright']['Eyeglasses'] = face['Eyeglasses']
            data_out['tiltright']['Sunglasses'] = face['Sunglasses']
            data_out['tiltright']['Gender'] = face['Gender']
            data_out['tiltright']['Beard'] = face['Beard']
            data_out['tiltright']['Mustache'] = face['Mustache']
            data_out['tiltright']['EyesOpen'] = face['EyesOpen']
            data_out['tiltright']['MouthOpen'] = face['MouthOpen']
            data_out['tiltright']['Emotions'] = face['Emotions']
            data_out['tiltright']['Quality'] = face['Quality']
        
        if(smile_flg == 'yes') : 
            face = datadict[ts_smile_face]
            data_out['smile']['AgeRange'] = {}
            data_out['smile']['Smile'] = {}
            data_out['smile']['Eyeglasses'] = {}
            data_out['smile']['Sunglasses'] = {}
            data_out['smile']['Gender'] = {}
            data_out['smile']['Beard'] = {}
            data_out['smile']['Mustache'] = {}
            data_out['smile']['EyesOpen'] = {}
            data_out['smile']['MouthOpen'] ={}
            data_out['smile']['Emotions'] =[]
            
            data_out['smile']['AgeRange'] = face['AgeRange']        
            data_out['smile']['Smile'] = face['Smile']
            data_out['smile']['Eyeglasses'] = face['Eyeglasses']
            data_out['smile']['Sunglasses'] = face['Sunglasses']
            data_out['smile']['Gender'] = face['Gender']
            data_out['smile']['Beard'] = face['Beard']
            data_out['smile']['Mustache'] = face['Mustache']
            data_out['smile']['EyesOpen'] = face['EyesOpen']
            data_out['smile']['MouthOpen'] = face['MouthOpen']
            data_out['smile']['Emotions'] = face['Emotions']
            data_out['smile']['Quality'] = face['Quality']
    
        
        if(blink_flg == 'yes') :  
            face = datadict[ts_blink_face]
            data_out['blink']['AgeRange'] = {}
            data_out['blink']['Smile'] = {}
            data_out['blink']['Eyeglasses'] = {}
            data_out['blink']['Sunglasses'] = {}
            data_out['blink']['Gender'] = {}
            data_out['blink']['Beard'] = {}
            data_out['blink']['Mustache'] = {}
            data_out['blink']['EyesOpen'] = {}
            data_out['blink']['MouthOpen'] ={}
            data_out['blink']['Emotions'] =[]
            
            data_out['blink']['AgeRange'] = face['AgeRange']        
            data_out['blink']['Smile'] = face['Smile']
            data_out['blink']['Eyeglasses'] = face['Eyeglasses']
            data_out['blink']['Sunglasses'] = face['Sunglasses']
            data_out['blink']['Gender'] = face['Gender']
            data_out['blink']['Beard'] = face['Beard']
            data_out['blink']['Mustache'] = face['Mustache']
            data_out['blink']['EyesOpen'] = face['EyesOpen']
            data_out['blink']['MouthOpen'] = face['MouthOpen']
            data_out['blink']['Emotions'] = face['Emotions']
            data_out['blink']['Quality'] = face['Quality']
         
            
        
        
        with open(output_json, 'w') as outfile:  
            json.dump(data_out, outfile)
            
        loginfo += "time taken by aws comparision:::: " +  str(time.time()-start) + '\n'
        #print(loginfo)
        logging.info(loginfo)
    return data_out
