import sys
import cv2 as cv
import histogram as hist
import pathlib, json
import xlsxwriter
import time
import _thread

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

IMAGE_PATH = ["data/jpeg", "data/png"]
INDEX_FILES_PATH = "data/index"

def findImage(imageToFind, worksheet, row):
    t0 = time.time()
    videoName = ''
    timeInVideo = 0
    minEucDistance = sys.maxsize
    IndexVidHistDir = pathlib.Path(INDEX_FILES_PATH)
    img = cv.imread(str(imageToFind))
    imgHist = hist.get_image_histograms(img)
    for videoHists in IndexVidHistDir.iterdir():
        with open(videoHists, 'r') as vidHistFile: 
            json_object = json.load(vidHistFile) 
        
        for frameInfo in json_object:
            frameHist = frameInfo['hist']
            eucDistanceFrameImage = hist.get_euclidien_distance(imgHist, frameHist)
            
            if(eucDistanceFrameImage < minEucDistance):
                minEucDistance = eucDistanceFrameImage
                timeInVideo = frameInfo['time']
                videoName = videoHists.stem
    
    execTime = time.time() - t0
    worksheet.write(row, 0, str(imageToFind.stem))
    worksheet.write(row, 1, str(videoName))
    worksheet.write(row, 2, timeInVideo)
    worksheet.write(row, 3, minEucDistance)
    worksheet.write(row, 4, execTime)

def main():
    results = xlsxwriter.Workbook('results.xlsx')
    
    for imgPath in IMAGE_PATH:
        print('Treating images in ' + str(imgPath))
        currentImgDir = pathlib.Path(imgPath)
        
        sheetName = str(imgPath).split("/",1)[1]
        worksheet = results.add_worksheet(sheetName)
        worksheet.write(0, 0, 'image') 
        worksheet.write(0, 1, 'video') 
        worksheet.write(0, 2, 'minutage')
        worksheet.write(0, 3, 'distance euclidienne')
        worksheet.write(0, 4, 'temps demandé') 
        row = 0
        
        for imageToFind in currentImgDir.iterdir():
            row += 1
            _thread.start_new_thread(findImage, (imageToFind, worksheet, row))
            # findImage(imageToFind, worksheet, row)
 
        print('Treatment of images in ' + str(imgPath) + ' is done')   
    results.close()
    quit()
    
if __name__ == '__main__':
    main()
