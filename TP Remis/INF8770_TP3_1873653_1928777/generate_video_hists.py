
import cv2 as cv
import hist
import pathlib, json
import time
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

VIDEO_PATH = "data/video"
INDEX_FILES_PATH = "data/index/"

def main():
    t0 = time.time()
    currentDirectory = pathlib.Path(VIDEO_PATH)
        
    for videoName in currentDirectory.iterdir():
        video = cv.VideoCapture(str(videoName))
        
        if not video.isOpened():
            print("Error - could not open video " + videoName)
            return
        else:
            print("Parsing video " + str(videoName) + "...")
        
        frameIndex = 0
        videoData = []
        fps = round(video.get(cv.CAP_PROP_FPS))
        
        while(True):          
            ret, frame = video.read()
            if not ret:
                break
                
            if (frameIndex % fps == 0):
                #cv.imshow('frame', frame)
                #cv.waitKey(1)
                
                videoHist = hist.get_image_histograms(frame)
                timeStamp = video.get(cv.CAP_PROP_POS_MSEC)/1000.0
                videoData.append({
                    'frameNumber' : frameIndex,
                    'time': timeStamp,
                    'hist' : videoHist.tolist() 
                })
                
            frameIndex += 1
          
        with open(INDEX_FILES_PATH + videoName.stem + ".txt", 'w') as outfile:
            json.dump(videoData, outfile)

        video.release()
        cv.destroyAllWindows()
    
    print("Time taken: " + (str(time.time() - t0)))
               
if __name__ == '__main__':
    main()
