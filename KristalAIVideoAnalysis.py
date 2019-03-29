# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import time
import logging

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return "Hello, Kristal AI Video Analysis APIs!"
    
# endpoint to create new user
@app.route("/matchID", methods=["POST"])
def matchid():
    video = request.json['video']
    sourceFile = request.json['kyc']
    jsonString = request.json['poses']
    video_fileName, vidext = video.split('.')[0], video.split('.')[1]
    fname = 'apilog.log'
    logging.basicConfig(filename=fname,filemode='w', level=logging.INFO)
    logging.info(jsonString)
    
    from matchID import VideoDetect
    st = time.time()
    analyzer=VideoDetect(video,sourceFile)
    if analyzer.validID == 1 :
        st1 = time.time()
        analyzer.main()
        logging.info("total time taken to recieve from aws rekognition ==  %s", str(time.time()-st1))
        rekognition_time = "total time taken by aws rekognition"+str(time.time()-st1)
        
        app.logger.info(rekognition_time)
        
        
        from generate_output import generate
        # mainfolder_path : /a/b/c/test/
        # video_pathIn :  /a/b/c/test/test.MOV
        # faceinfo_jsonFile : /a/b/c/test/test.json
        # video_fileName : test
        # sourceFile : /d/e/f/PAN.jpg
        data_out = generate(analyzer.mainfolder_path,analyzer.video_pathIn,analyzer.faceinfo_jsonFile,analyzer.video_fileName,analyzer.sourceFile,jsonString)
        
        total_time = "total time taken by whole process"+ str(time.time()-st)
        app.logger.info(total_time)
        logging.info(total_time)
        
        return jsonify(data_out)
    else:
        data_out = "No photo present/poor quality images in ID proof"
        return jsonify(data_out)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
    # app.run(host="0.0.0.0", port=80,  debug=True)
    
