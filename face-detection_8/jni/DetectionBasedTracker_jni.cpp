#include <DetectionBasedTracker_jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/detection_based_tracker.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "img.cpp";
#include <android/log.h>

#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;
/**扩充**/
#define CLOSE 1
#define OPEN 0
#define OFFSET -20
#define STEP 8		//平滑步长
#define STEP_F 5	//寻找局部峰值步长
IplImage* painty;
IplImage* painty2;
int WITH=90,HEIGHT=60;
int x,y;
CvScalar s,t;
int summit_num = 0;//波峰数
int k;
int summit_1=0;//峰值坐标1
int summit_2=0;//峰值坐标2
int check;//检测点
int temp = 0;

img arr_img[100];
vector<vector<Point> > contours;

int trs = 0;//像素值发生改变标志
IplImage* bin_nom_eye;
/*扩充结束**/
void sort_img();
void exchange(img * p1,img * p2);

inline int otsu(IplImage* src)//大津算法（最大类间差算法）
{
	int height;
	int width;
	int i,j;
	unsigned char*p;
	float histogram[256] = {0};
	int size;
	float avgValue=0;
	int threshold;
	float maxVariance=0;
	float w = 0, u = 0;
	float t;
	float variance;

	height = src->height;
	width = src->width;
	//histogram
	for(i=0; i < height; i++)
	{
		p = (unsigned char*)src->imageData + src->widthStep * i;
		for(j = 0; j < width; j++)
		{
			histogram[*p++]++;
		}
	}
	//normalize histogram
	size = height * width;
	for(i = 0; i < 256; i++)
	{
		histogram[i] = histogram[i] / size;
	}

	//average pixel value
	for(i=0; i < 256; i++)
	{
		avgValue += i * histogram[i];  //整幅图像的平均灰度
	}

	for(i = 0; i < 256; i++)
	{
		w += histogram[i];
		u += i * histogram[i];
		t = avgValue * w - u;
		variance = t * t / (w * (1 - w) );
		if(variance > maxVariance)
		{
			maxVariance = variance;
			threshold = i;
		}
	}
	return threshold+OFFSET;
}

inline void smooth(int a[], int n)//平滑、寻找峰值
{
	int start = STEP - STEP/2;
	int end = n - STEP/2 ;
	double temp = 0;
	int i,j;

        int res[256] = {0};

	for (i=start; i<end; i++)
	{
		temp = 0;
		for (j=0-STEP/2; j<STEP/2; j++)
			temp += a[i + j];
                temp /= STEP;
                res[i] = (int)temp;
	}
	for (i=0; i<n; i++)
		a[i] = res[i];
//find summit
	for (i=0; i<n; i++)
		res[i]=0;
	for (i=start; i<end; i++)
		for(j=i-STEP_F/2; j<=i+STEP_F/2; j++)
		{
			if(a[i]<a[j])
			{
				res[i]=0;
				break;
			}
			else
				res[i]=a[i];
		}
	for (i=0; i<n; i++)
	{
		if(res[i] != res[i-1])
			a[i]=res[i];
	}

	return;
}


inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

JNIEXPORT jlong JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject
(JNIEnv * jenv, jclass, jstring jFileName, jint faceSize)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject enter");
    const char* jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    jlong result = 0;

    try
    {
        DetectionBasedTracker::Parameters DetectorParams;
        if (faceSize > 0)
            DetectorParams.minObjectSize = faceSize;
        result = (jlong)new DetectionBasedTracker(stdFileName, DetectorParams);
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeCreateObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeCreateObject()");
        return 0;
    }

    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject exit");
    return result;
}

JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject
(JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject enter");
    try
    {
        if(thiz != 0)
        {
            ((DetectionBasedTracker*)thiz)->stop();
            delete (DetectionBasedTracker*)thiz;
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeestroyObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDestroyObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeDestroyObject()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject exit");
}

JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart
(JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart enter");
    try
    {
        ((DetectionBasedTracker*)thiz)->run();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStart caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStart caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStart()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart exit");
}

JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop
(JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop enter");
    try
    {
        ((DetectionBasedTracker*)thiz)->stop();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStop caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStop()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop exit");
}

JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize
(JNIEnv * jenv, jclass, jlong thiz, jint faceSize)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize enter");
    try
    {
        if (faceSize > 0)
        {
            DetectionBasedTracker::Parameters DetectorParams = \
            ((DetectionBasedTracker*)thiz)->getParameters();
            DetectorParams.minObjectSize = faceSize;
            ((DetectionBasedTracker*)thiz)->setParameters(DetectorParams);
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeSetFaceSize caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeSetFaceSize()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize exit");
}


JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect
(JNIEnv * jenv, jclass, jlong thiz, jlong imageGray, jlong faces)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect enter");
    try
    {
        vector<Rect> RectFaces;
        ((DetectionBasedTracker*)thiz)->process(*((Mat*)imageGray));
        ((DetectionBasedTracker*)thiz)->getObjects(RectFaces);
        vector_Rect_to_Mat(RectFaces, *((Mat*)faces));
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect exit");
}

JNIEXPORT jint JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeJudge
(JNIEnv * jenv, jclass, jlong thiz, jlong imageGray,jint type)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect enter");
    try
    {	int threshod;
        jint status;
        jint max_m;
        int max=0,cur=0;
    	IplImage ipl_img(*(Mat*)imageGray);
    	 threshod=otsu(&ipl_img);
    	 WITH=ipl_img.width;
    	 HEIGHT=ipl_img.height;
    	 int DEPTH=ipl_img.depth;
    	 int CHANNEL=ipl_img.nChannels;
    	 int h_acc[HEIGHT];
    	 bin_nom_eye = cvCreateImage(cvSize(WITH,HEIGHT),DEPTH, CHANNEL);
    	cvThreshold(&ipl_img, bin_nom_eye,threshod, 255, CV_THRESH_BINARY);
		painty = cvCreateImage(cvSize(WITH,HEIGHT),DEPTH, CHANNEL);
		cvZero(painty);
		memset(h_acc, 0, HEIGHT*4);
		for(y=0; y<HEIGHT; y++)
			for(x=0; x<WITH; x++)
			{
				s = cvGet2D(bin_nom_eye,y, x);
				if(s.val[0] == 0)
					h_acc[y]++;
			}
		for(y=0; y<HEIGHT; y++)
		{
			if(h_acc[y]>8){
					cur++;
					}else{
					if(max<cur)
					max=cur;
					cur=0;
					}
		}
		if(max<cur)
		max=cur;
		smooth(h_acc, HEIGHT);
		if(type==2){
		/*painty2 = cvCreateImage(cvSize(WITH+3,HEIGHT+3),DEPTH, CHANNEL);
		cvZero(painty2);
		for(y=0; y<HEIGHT; y++)
			for(x=0; x<h_acc[y]; x++)
			{
				t.val[0] = 255;
				cvSet2D(painty2, y, x, t);
			}
*/
		//检测波峰或波峰间二值图是否有空洞
		for(k=0; k<HEIGHT; k++)
		{
			if(h_acc[k]!=0)
			{
				summit_num++;
				if(h_acc[k]>summit_1 && h_acc[k]>summit_2)
				{
					summit_2 = summit_1;
					summit_1 = k;
				}
				else if(h_acc[k]>summit_2)
					summit_2 = k;
			}
		}
		if(summit_num == 1)
			check = summit_1;
		else if(summit_num >= 2)
			check = (summit_1 + summit_2)/2;
		trs = 0;
		s = cvGet2D(bin_nom_eye, check, 1);
		temp = s.val[0];
		for(x=0; x<WITH; x++)
		{
			s = cvGet2D(bin_nom_eye, check, x);
			if(s.val[0]==0 && temp==255)
				trs++;
			else if(trs==1 && s.val[0]==255 && temp==0 )
				trs++;
			temp = s.val[0];
		}
		if(trs >= 2)
			status = 1;
		else
			status = 0;
		return status;
		}else if(type==1){
			max_m=max;
			return max_m;
		}
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect exit");
}


JNIEXPORT jint JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeJudgem
(JNIEnv * jenv, jclass, jlong thiz, jlong imageGray)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect enter");
    try
    {	int threshod;
        jint status;
        Mat canny_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        IplImage ipl_img(*(Mat*)imageGray);
        WITH=ipl_img.width;
        HEIGHT=ipl_img.height;
        int DEPTH=ipl_img.depth;
        int CHANNEL=ipl_img.nChannels;
   	 	bin_nom_eye = cvCreateImage(cvSize(WITH,HEIGHT),DEPTH, CHANNEL);
   	 	 //cvThreshold(&ipl_img, bin_nom_eye,threshod, 255, CV_THRESH_BINARY);
   		int max_thresh = 255;
   		int max_y=0;
   		int min_y=255;

    	 threshod=otsu(&ipl_img);
    	 //Mat  matImg = Mat(ipl_img,true);
    	 cvtColor( *(Mat*)imageGray, *(Mat*)imageGray, CV_BGR2GRAY );
         blur(*(Mat*)imageGray,*(Mat*)imageGray, Size(3,3) );
    	 Canny( *(Mat*)imageGray,canny_output, threshod, threshod * 2, 3);
    	 findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    	 for( int i = 0; i< contours.size(); i++ )
    	 {
    	   arr_img[i].id=i;
    	   arr_img[i].nums=contours.at(i).size();
    	 }
    	 sort_img();
    	 for( int i = 0; i< contours.size(); i++ )
    	 {
    	 	if(contours.at(i).size() == arr_img[0].nums||contours.at(i).size() == arr_img[1].nums){
    	 	for(int j=0;j<contours.at(i).size();j++){
    	 		Point p = contours.at(i).at(j);
    	 		if(p.y>max_y){
    	 		max_y=p.y;
    	 		}if(p.y<min_y){
    	 		min_y=p.y;
    	 		}
    	 	}
    	 	}
    	 }
    	 return (max_y-min_y);
    	// return  max_y;
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect exit");
}

void sort_img(){
	for(int i=0;i<contours.size();i++){
		for(int j=1;j<contours.size()-i;j++){
			if(arr_img[j-1].nums<arr_img[j].nums)
				 exchange(&arr_img[j-1],&arr_img[j]);
		}
	}
}

void exchange(img * p1,img * p2){
	int id_;
	int nums_;
	id_ = p1->id;
	nums_=p1->nums;
	p1->id=p2->id;
	p1->nums=p2->nums;
	p2->id=id_;
	p2->nums=nums_;
}
