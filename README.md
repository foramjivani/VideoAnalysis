# MLSearce

KristalAIVideoAnalysis.py <br />
This file is the main entry point for API. If one want to see all the messeages or error encountered then can write debug = True in main function under app.run funaction. Moreover can change port number also.

matchID.py <br />
This file generates json file which contains all face related information obtained via amazon face rekognition.
If one want to debug then one can save all target images, csv file related to pose, smile and blink in server itself by uncommenting below line: <br />
#dirPath  = os.path.dirname(os.path.realpath(__file__)) + '/' <br />

and commenting below line for detailed information: <br />
dirPath = tempfile.mkdtemp() <br /> 

generate_output.py <br /> 
This file contains necessary compuation for getting pose frame as target to compare with kyc document for each pose mentioned in request and stores them in s3 target bucket as zip file. <br />
This generates output json which gives whether document is matched with target images or not. <br />

apilog.log <br />
This is empty file. <br />
This file stores how much time each step is taking,total time taken by whole process and other information.
