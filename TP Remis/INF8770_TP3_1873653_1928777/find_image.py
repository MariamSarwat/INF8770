import sys
import cv2 as cv
import hist
import pathlib, json
import xlsxwriter
import time

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

IMAGE_PATH = ["data/jpeg", "data/png"]
INDEX_FILES_PATH = "data/index"

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
        row = 1
        
        for imageToFind in currentImgDir.iterdir():
            t0 = time.time()
            img = cv.imread(str(imageToFind))
            imgHist = hist.get_image_histograms(img)
            
            IndexVidHistDir = pathlib.Path(INDEX_FILES_PATH)
            minEucDistance = sys.maxsize
            videoName = ''
            timeInVideo = 0
            
            for videoHists in IndexVidHistDir.iterdir():
                with open(videoHists, 'r') as vidHistFile: 
                    json_object = json.load(vidHistFile) 
                
                for frameInfo in json_object:
                    frameHist = frameInfo['hist']
                    eucDistanceFrameImage = hist.get_euclidean_distance(imgHist, frameHist)
                    
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
            row += 1
            
        print('Treatment of images in ' + str(imgPath) + ' is done')   
    results.close()

    
if __name__ == '__main__':
    main()
