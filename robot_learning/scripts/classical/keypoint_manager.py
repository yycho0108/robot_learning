import numpy as np
import cv2

#void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 
#
#//this function automatically gets rid of points for which tracking fails
#
#  vector<float> err;					
#  Size winSize=Size(21,21);																								
#  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
#
#  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
#
#  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
#  int indexCorrection = 0;
#  for( int i=0; i<status.size(); i++)
#     {  Point2f pt = points2.at(i- indexCorrection);
#     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
#     		  if((pt.x<0)||(pt.y<0))	{
#     		  	status.at(i) = 0;
#     		  }
#     		  points1.erase (points1.begin() + i - indexCorrection);
#     		  points2.erase (points2.begin() + i - indexCorrection);
#     		  indexCorrection++;
#     	}
#     }
#}

class KeypointManager(object):
    def __init__(self):
        self.kpts_ = []
    def __call__(self, img, kpts):
        if len(self.kpts_) <= 0:
            self.kpts_ = kpts
            return

