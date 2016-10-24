package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.Math.*;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;
   // private static final int  NUM_OF_FACES  = 1;
    
    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem				mItemFace10;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private Mat                    teplateR;
    private Mat                    teplateL;
    private Mat                    teplateM;
    private Mat                    mZoomWindow;
    private Mat                    mZoomWindow2;
    private Mat                    mZoomWindow1;
    private File                   mCascadeFile;

    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;
    private CascadeClassifier      mJavaDetectorEyeL;
    private CascadeClassifier      mJavaDetectorEyeR;
    private CascadeClassifier      mJavaDetectorMouth;
    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;
    private int status_R;
    private int status_L;
    private int status_M;
    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;
    double xCenter = -1;
    double yCenter = -1;
    private CameraBridgeViewBase   mOpenCvCameraView;
    private double eyel_x;
    private double eyel_y;
    private double eyer_x;
    private double eyer_y;
    private double distance1=0.0;
    private double distance2=0.0;
    private double width_m=0;
    private double height_m=0;
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                       // mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }

                        is.close();
                        os.close();
                        /*
                         * 加载右眼分类器
                         */
                         InputStream iser = getResources().openRawResource(
                         R.raw.haarcascade_righteye_2splits);
                         File cascadeDirER = getDir("cascadeER",
                         Context.MODE_PRIVATE);
                         File cascadeFileER = new File(cascadeDirER,
                         "haarcascade_righteye_2splits.xml");
                         FileOutputStream oser = new FileOutputStream(cascadeFileER);
                         /*
                         * 定义一个足够大的byte类型的数组，存放加载右眼部分类器文件的数据
                         */
                         byte[] bufferER = new byte[4096];
                         int bytesReadER;
                         while ((bytesReadER = iser.read(bufferER)) != -1) {
                         oser.write(bufferER, 0, bytesReadER);
                         }
                         iser.close();
                         oser.close();
                         /*
                          * 加载左眼分类器
                          */
                          InputStream isel = getResources().openRawResource(
                          R.raw.haarcascade_lefteye_2splits);
                          File cascadeDirEL = getDir("cascadeEL",
                          Context.MODE_PRIVATE);
                          File cascadeFileEL = new File(cascadeDirEL,
                          "haarcascade_lefteye_2splits.xml");
                          FileOutputStream osel = new FileOutputStream(cascadeFileEL);
                          /*
                          * 定义一个足够大的byte类型的数组，存放加载左眼部分类器文件的数据
                          */
                          byte[] bufferEL = new byte[4096];
                          int bytesReadEL;
                          while ((bytesReadEL = isel.read(bufferEL)) != -1) {
                          osel.write(bufferEL, 0, bytesReadEL);
                          }
                          isel.close();
                          osel.close();
                          /**加载脸部检测*/
                          InputStream ism = getResources().openRawResource(R.raw.haarcascade_mcs_mouth);
                          File cascadeDirM = getDir("cascadeM",
                          Context.MODE_PRIVATE);
                          File cascadeFileM = new File(cascadeDirM,
                          "haarcascade_mcs_mouth.xml");
                          FileOutputStream osm = new FileOutputStream(cascadeFileM);
                          byte[] bufferM = new byte[4096];
                          int bytesReadM;
                          while ((bytesReadM = ism.read(bufferM)) != -1) {
                          osm.write(bufferM, 0, bytesReadM);
                          }
                          ism.close();
                          osm.close();
                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                        /**
                         * opencv的迭代分类器可以持续迭代，将分类器迭代为强分类器，这样就可以从集中一张脸到
                         集中到眼睛上面，可以实现对人眼的识别与检测； 这里是对右眼进行检测
                         */
                         mJavaDetectorEyeR = new CascadeClassifier(
                         cascadeFileER.getAbsolutePath());
                         if (mJavaDetectorEyeR.empty()) {
                         Log.e(TAG, "Failed to load cascade classifier");
                         mJavaDetectorEyeR = null;
                         } else
                         Log.i(TAG, "Loaded cascade classifier from "
                         + mCascadeFile.getAbsolutePath());
                         cascadeDirER.delete();
                         /**
                         * 这里是对左眼进行迭代与检测
                         */
                         mJavaDetectorEyeL = new CascadeClassifier(
                         cascadeFileEL.getAbsolutePath());
                         if (mJavaDetectorEyeL.empty()) {
                         Log.e(TAG, "Failed to load cascade classifier");
                         mJavaDetectorEyeL = null;
                         } else
                         Log.i(TAG, "Loaded cascade classifier from "
                         + mCascadeFile.getAbsolutePath());
                         cascadeDirEL.delete();
                         //mouth detector
                         mJavaDetectorMouth=new CascadeClassifier(
                         	    cascadeFileM.getAbsolutePath());
                         if (mJavaDetectorMouth.empty()) {
                             Log.e(TAG, "Failed to load cascade classifier");
                             mJavaDetectorMouth = null;
                             }  else
                                 Log.i(TAG, "Loaded cascade classifier from "
                                 	    + mCascadeFile.getAbsolutePath());
                                       cascadeDirM.delete();
                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
                        cascadeDir.delete();
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.setCameraIndex(0);//TODO
                    /**
                    * enableFpsMeter()方法确定在屏幕上fps值的标签，其中FPS：Frame Per Second (每秒
                    帧数)
                    */
                    mOpenCvCameraView.enableFpsMeter();
                    /**
                    * enableView()方法建立Camera连接
                    */
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.face_detect_surface_view);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        CreateAuxiliaryMats();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mZoomWindow.release();
        mZoomWindow2.release();
        mZoomWindow1.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }
        if (mZoomWindow == null || mZoomWindow2 == null||mZoomWindow1==null)
            CreateAuxiliaryMats();
        MatOfRect faces = new MatOfRect();
        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null){
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
                }
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        if( facesArray.length>0)
        {
        Core.rectangle(mRgba, facesArray[0].tl(), facesArray[0].br(),
        FACE_RECT_COLOR, 3);
        xCenter = (facesArray[0].x + facesArray[0].width + facesArray[0].x) / 2;
        yCenter = (facesArray[0].y + facesArray[0].y + facesArray[0].height) / 2;
        Point center = new Point(xCenter, yCenter);
        Core.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);
        //Core.putText(mRgba, "[" + center.x + "," + center.y + "]",
       // new Point(center.x + 20, center.y + 20),
       // Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));
        Rect r = facesArray[0];
        // compute the eye area
        // split it
        Rect eyearea_right = new Rect(r.x + r.width / 16,
        (int) (r.y + (r.height / 4.5)), (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
        Rect eyearea_left = new Rect(r.x + r.width / 16+ (r.width - 2 * r.width / 16) / 2,
        (int) (r.y + (r.height / 4.5)),(r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
       // Rect mouth_area = new Rect(r.x + r.width / 2- r.width /5,(int) (r.y + (r.height / 18)*13),(r.width/3+r.width/10) , (int) (r.height / 4.0));
        Rect mouth_area = new Rect(r.x + r.width / 2- r.width /5,(int) (r.y + r.height-r.height*9/30),
        		(r.width/3+r.width/10) , (int) (r.height / 4.0));
        // draw the area - mGray is working grayscale mat, if you want to
        // see area in rgb preview, change mGray to mRgba
        Core.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
        new Scalar(255, 0, 0, 255), 2);
        Core.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
        new Scalar(255, 0, 0, 255), 2);
        Core.rectangle(mRgba, mouth_area.tl(), mouth_area.br(),
         new Scalar(255, 0, 0, 255), 2);
        teplateR = get_template(mJavaDetectorEyeR, eyearea_right, 24,0);
        teplateL = get_template(mJavaDetectorEyeL, eyearea_left, 24,1);
        distance1=Math.sqrt((eyel_x-eyer_x)*(eyel_x-eyer_x)+(eyel_y-eyer_y)*(eyel_y-eyer_y));
       if(distance2!=0.0){
        	if(distance2<=0.5*distance1)
                Core.putText(mRgba, "HEI   stay focus !!!",new Point(center.x + 20, center.y + 20),
                Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));
        }
        distance2=distance1;
        
       /* Core.putText(mRgba,distance1+"!!!" ,new Point(center.x + 20, center.y + 20),
                Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));
        */
        teplateM= get_template(mJavaDetectorMouth, mouth_area, 60,2);   
        
        if(!teplateR.empty())
        	status_R=mNativeDetector.judge(teplateR,2);//TODO
        Core.putText(mRgba, "teplateR:"+status_R+"!!!!!!!!!!!", new Point(center.x -100, center.y - 20), 
        		Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(100, 0, 0, 100));
        if(!teplateL.empty())
        	status_L=mNativeDetector.judge(teplateL,2);
        Core.putText(mRgba, "teplateL:"+status_L+"!!!!!!!!!!!", new Point(center.x +100, center.y -20), 
        		Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(100, 0, 0, 100));
        if(!teplateM.empty())
        	status_M=mNativeDetector.judge(teplateM,1);
        	if(status_M<=0.23*teplateM.cols())
        		status_M=0;
        	else
        		status_M=1;
        Core.putText(mRgba, "teplateM:"+status_M+"!!!!!!!!!!!", new Point(center.x , center.y +100), 
        		Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(100, 100, 100, 100));
        }
 
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemFace10 = menu.add("Face size 10%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemFace10)
        	setMinFaceSize(0.1f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }
    private void CreateAuxiliaryMats() {
    if (mGray.empty())
    return;
    int rows = mGray.rows();
    int cols = mGray.cols();
    if (mZoomWindow == null) {
    //mZoomWindow = mRgba.submat(rows / 2 + rows / 10, rows, cols / 2+ cols / 10, cols);
    mZoomWindow = mRgba.submat(0, rows / 2 - rows / 8, 0, cols / 2-cols / 8);
    mZoomWindow2 = mRgba.submat(0, rows / 2 - rows / 8, cols / 2+ cols / 8, cols);
    }
    if( mZoomWindow1==null){
        mZoomWindow1 = mRgba.submat((rows/3)*2, rows , cols /3, 4*cols / 6);
    }
    }
    
    private Mat get_template(CascadeClassifier clasificator, Rect area, int size,int type) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        if(type==1||type==0)
        clasificator.detectMultiScale(mROI, eyes, 1.1, 2,
        Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(50, 50),
        new Size());
        else
        	 clasificator.detectMultiScale(mROI, eyes, 1.1, 2,Objdetect.CASCADE_FIND_BIGGEST_OBJECT, new Size(30, 30),new Size());
        Rect[] eyesArray = eyes.toArray();
        if(eyesArray.length>0){
        Rect e = eyesArray[0];
        e.x = area.x + e.x;
        e.y = area.y + e.y;
        		    if(type==1||type==0){
        		    Rect eye_only_rectangle = new Rect((int) e.tl().x,
        		    (int) (e.tl().y + e.height * 0.4), (int) e.width,
        		    (int) (e.height * 0.6));
        		    mROI = mGray.submat(eye_only_rectangle);
        		    Mat vyrez = mRgba.submat(eye_only_rectangle);
        		    Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);
        		    Core.circle(vyrez, mmG.minLoc, 4, new Scalar(255, 255, 255, 255), 2);
        		    if(type==1)
        		    {eyel_x=mmG.minLoc.x;
        		      eyel_y=mmG.minLoc.y;
        		    }if(type==0){
        		      eyer_x=mmG.minLoc.x;
          		      eyer_y=mmG.minLoc.y;
        		    }
        		    iris.x = mmG.minLoc.x + eye_only_rectangle.x;
        		    iris.y = mmG.minLoc.y + eye_only_rectangle.y;
        		    eye_template = new Rect((int) iris.x - size / 2, (int) iris.y- size / 2, size, size);
        		    Core.rectangle(mRgba, eye_template.tl(), eye_template.br(),new Scalar(0, 0, 0, 0), 1);
        		    template = (mGray.submat(eye_template)).clone();//TODO
        		    return template;
        }
        if(type==2){
        	
        	//Rect mouth_only_rectangle = new Rect((int) (e.tl().x+(e.width-(e.height+e.width)/2)/2),(int) (e.tl().y ), (int) ((e.height+e.width)/2),(int) ((e.height+e.width)/2));
        	Rect mouth_only_rectangle = new Rect((int) (e.tl().x+e.width/10),(int) (e.tl().y ), (int) ((e.height*0.3+e.width*0.7)),(int) ((e.height*0.3+e.width*0.7)));
		    Core.rectangle(mRgba, mouth_only_rectangle.tl(), mouth_only_rectangle.br(),new Scalar(0, 0, 0, 0), 1);
		    width_m=e.width;
		    height_m=e.height;
		    template = (mRgba.submat(mouth_only_rectangle )).clone();
		    return template;
        }
        }
        return template;
        }   
    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
