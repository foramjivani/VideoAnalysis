# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:45:51 2019

@author: foram.jivani
"""

import boto3
import json
import sys

import time
import botocore

import pathlib
import tempfile
import os
import logging

class VideoDetect:
    
    def __init__(self, video, sourceFile):
        self.video = video
        self.sourceFile = sourceFile
        self.loginfo = ''
        ##############    path where all the pose images and target will be stored        ###################
        #dirPath  = os.path.dirname(os.path.realpath(__file__)) + '/'  #/a/b/c/
        dirPath = tempfile.mkdtemp()
        
        
        self.video_fileName, vidext = video.split('.')[0], video.split('.')[1]
        self.source_fileName, imgext = sourceFile.split('.')[0], sourceFile.split('.')[1]
        
        rootFolderPath = dirPath   # /a/b/c/Script/
        #print(rootFolderPath) # /a/b/c/Script/
        #################          create folder          ####################
        self.mainfolder_path = rootFolderPath + self.video_fileName  # /a/b/c/Script/test
        pathlib.Path(self.mainfolder_path).mkdir(parents=True, exist_ok=True) 
        self.mainfolder_path = rootFolderPath + self.video_fileName + '/'  # /a/b/c/Script/test/
        
        ###############          path where we store downloaded video and Image      ####################
        self.video_pathIn = self.mainfolder_path + video  #/a/b/c/Script/test/test.MOV
        self.image_pathIn = self.mainfolder_path + sourceFile
        
        ##############        get faces json file      #####################
        self.faceinfo_jsonFile = self.mainfolder_path + self.video_fileName +".json"  # /a/b/c/Script/test/test.json

        self.jobId = ''
        self.rek = boto3.client('rekognition') 
        self.queueUrl = 'https://sqs.us-east-1.amazonaws.com/348221620929/KristalAIDemoQueue'
        self.roleArn = 'arn:aws:iam::348221620929:role/kristal-staging-rekognition-access'
        self.topicArn = 'arn:aws:sns:us-east-1:348221620929:AmazonRekognition_Kristal_AI_Demo'
        self.videobucket = 'kristal-ai-demo'
        self.imagebucket = 'kristal-ai-demo-idproofs'
        self.validID = 1
        
        s3 = boto3.resource('s3')
    
        try:
            st = time.time()
            s3.Bucket(self.imagebucket).download_file(sourceFile, self.image_pathIn)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                self.loginfo += "Image  object does not exist." + '\n'
            else:
                raise
        
        self.loginfo += '#######################################' +'\n'
        self.loginfo += 'Image downloaded from s3 bucket at location = '+ str(self.image_pathIn) +'\n'
        self.loginfo += 'time taken to download Image = ' + str(time.time()-st) +'\n'
        self.loginfo += '#######################################' + '\n'
        
        with open(self.image_pathIn, 'rb') as image:
            st = time.time()
            response = self.rek.detect_faces(Image={'Bytes': image.read()})
        
        if len(response['FaceDetails']  ) !=0 :
            self.loginfo += 'Time taken to detect face in ID proof ===== '+ str(time.time()-st)+'\n'
        else:
            self.loginfo += 'No photo present/poor quality images in ID proof' + '\n'
            self.validID = 0
        
        #print(validID)   
        
        if(self.validID == 1):
            s3 = boto3.resource('s3')
            st = time.time()
            try:
                s3.Bucket(self.videobucket).download_file(video, self.video_pathIn)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    self.loginfo += "Video object does not exist." +'\n'
                else:
                    raise
            
            self.loginfo += '#######################################' +'\n'
            self.loginfo += 'video downloaded from s3 bucket at location ='+ str(self.video_pathIn)+'\n'
            self.loginfo += 'time taken to download video = ' + str(time.time()-st) + '\n'
            self.loginfo += '#######################################' +'\n'

    def main(self):

        jobFound = False
        sqs = boto3.client('sqs') 
        
        ############### start face detection #################
        response = self.rek.start_face_detection(Video={'S3Object':{'Bucket':self.videobucket,'Name':self.video}},
            NotificationChannel={'RoleArn':self.roleArn, 'SNSTopicArn':self.topicArn}, FaceAttributes='ALL') 
        ##########################
        
        self.loginfo += 'Start Job Id: ' + response['JobId']
        dotLine=0
        while jobFound == False:
            sqsResponse = sqs.receive_message(QueueUrl=self.queueUrl, MessageAttributeNames=['ALL'],
                                          MaxNumberOfMessages=10)

            if sqsResponse:
                
                if 'Messages' not in sqsResponse:
                    if dotLine<20:
                        print('.', end='')
                        dotLine=dotLine+1
                    else:
                        print()
                        dotLine=0    
                    sys.stdout.flush()
                    continue

                for message in sqsResponse['Messages']:
                    notification = json.loads(message['Body'])
                    rekMessage = json.loads(notification['Message'])
                    print(rekMessage['JobId'])
                    print(rekMessage['Status'])
                    if str(rekMessage['JobId']) == response['JobId']:
                        #print('Matching Job Found:' + rekMessage['JobId'])
                        jobFound = True
                        
                        ########### get face detection
                        
                        self.GetResultsFaces(rekMessage['JobId'])
                        ################

                        sqs.delete_message(QueueUrl=self.queueUrl,
                                       ReceiptHandle=message['ReceiptHandle'])
                    else:
                        print()
                        print("Job didn't match:" +
                              str(rekMessage['JobId']) + ' : ' + str(response['JobId']))
                    # Delete the unknown message. Consider sending to dead letter queue
                    sqs.delete_message(QueueUrl=self.queueUrl,
                                   ReceiptHandle=message['ReceiptHandle'])

        logging.info(self.loginfo)

    def GetResultsFaces(self, jobId):
        paginationToken = ''
        finished = False
        output_data={}
        output_data['Faces']=[]

        while finished == False:
            response = self.rek.get_face_detection(JobId=jobId,
                                            NextToken=paginationToken)
            
            for faceDetection in response['Faces']:
                output_data['Faces'].append(faceDetection)

            
            if 'NextToken' in response:
                paginationToken = response['NextToken']
            else:
                finished = True
        
        with open(self.faceinfo_jsonFile, 'w') as outfile:  
            json.dump(output_data, outfile)
