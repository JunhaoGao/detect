#include <iostream>  
#include <io.h>
#include <direct.h>
#include <fstream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <ctime> 


using namespace std;
using namespace cv;

bool TRAIN = false;   //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��  
bool CENTRAL_CROP = false;   //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����  
//int TRAINTYPE = 0;


//�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������  
//��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������  
class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����  
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����  
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

class myRect
{
public:
	string group;
	double w;
	Rect rect;
};
void generateDescriptors(ifstream& imagePath, HOGDescriptor& hog, vector<float>& descriptors, int& descriptorDim, 
	Mat& sampleFeatureMat, Mat& sampleLabelMat, int trainClass,int PosSamNO,int NegSamNO,int HardExampleNO) {
	string imgName;
	int numLimit;
	if (0 == trainClass)
	{
		numLimit = PosSamNO;
	}
	else if (1 == trainClass)
	{
		numLimit = NegSamNO;
	}
	else if (2 == trainClass)
	{
		numLimit = HardExampleNO;
	}
	for (int num = 0; num < numLimit && getline(imagePath, imgName); num++)
	{
		//cout << imgName << endl;
		Mat src = imread(imgName);//��ȡͼƬ  

		if (CENTRAL_CROP)
			resize(src, src, hog.winSize);
			//src = src(rectCrop);//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������  
								/*		imshow("....", src);
								waitKey(6000);			*/							 //resize(src,src,Size(64,128));  
		hog.compute(src, descriptors, hog.blockStride);//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
												  //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������  
												  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
		if (0 == trainClass)
		{
			if (0 == num)
			{
				descriptorDim = descriptors.size();	//HOG�����ӵ�ά�� 
													//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat  
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, descriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����  
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}
		else if (1 == trainClass) {
			if (0 == num)
				descriptorDim = sampleFeatureMat.cols;
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ1������
		}
		else if (2 == trainClass)
		{
			if (0 == num)
				descriptorDim = sampleFeatureMat.cols;
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
			sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ1������
		}

	}
	descriptors.clear();
	return;
}

void trainSVM(string posPath,string negPath, string hardPath, HOGDescriptor& hog, string modelPath, vector<float>& descriptors, int PosSamNO, int NegSamNO, int HardExampleNO) {

	ifstream finPos(posPath.data());
	ifstream finNeg(negPath.data());
	ifstream finHard(hardPath.data());
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������  
	MySVM svm;//SVM������
	//HOG����������
	string ImgName;//ͼƬ��(����·��) 
	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��      
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����  

	cout << "��ʼ���������������" << endl;
	generateDescriptors(finPos, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 0, PosSamNO, NegSamNO, HardExampleNO);
	cout << "�������" << endl;
	cout << "��ʼ���㸺���������" << endl;
	generateDescriptors(finNeg, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 1, PosSamNO, NegSamNO, HardExampleNO);
	cout << "�������" << endl;
	if (HardExampleNO > 0)
		//���ζ�ȡHardExample������ͼƬ������HOG������  
		generateDescriptors(finHard, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 2, PosSamNO, NegSamNO, HardExampleNO);
	
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01  
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	cout << "��ʼѵ��SVM������" << endl;
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������  
	cout << "ѵ�����" << endl;
	svm.save(modelPath.data());//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ� 
	descriptors.clear();
	finPos.close();
	finNeg.close();
	finHard.close();
	return;
}
	/*******************************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()��
	���Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	********************************************************************************************************************/
void setDetector(MySVM& svm, vector<float>& myDetector, string detectorPath){
	int DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��  
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���  
														  //cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������  
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������  
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��  

														   //��֧�����������ݸ��Ƶ�supportVectorMat������  
	for (int i = 0; i < supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��  
		for (int j = 0; j < DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";  
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��  
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��  
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�  
	resultMat = -1 * alphaMat * supportVectorMat;

	//��resultMat�е����ݸ��Ƶ�����myDetector��  
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������  
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;

	//�������Ӳ������ļ�  
	ofstream fout(detectorPath.data());
	for (int i = 0; i < myDetector.size(); i++)
		fout << myDetector[i] << endl;
	fout.close();
	
	return;
}

void DetectAndDraw(Mat& src, Mat &trtd, HOGDescriptor& hog1, HOGDescriptor& hog2, HOGDescriptor& hog3, vector<myRect>& found,
	vector<Rect>& found_tmp, vector<myRect>& found_filtered, vector<double>& weight){
	//string path
	//��ͼƬ���ж�߶����˼��
	string dirPath = "D:\\detectProject\\testdata\\";
	Rect r;
	myRect mr;
	hog1.detectMultiScale(src(Range(140, 300), Range(0, 480)), found_tmp, weight, 0.1, hog1.blockStride, Size(0, 0),1.05,2,false);
	//0.05~~0.1
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		if (r.tl().y <= 190 && r.br().y >= 200)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "small";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(),found_tmp.begin(),found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog2.detectMultiScale(src(Range(140, 300), Range(0, 480)), found_tmp, weight, 0.12, hog2.blockStride, Size(0, 0), 1.05, 2);
	//0.1~~0.15
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		if (r.tl().y <= 190 && r.br().y >= 215)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "middle";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	////found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog3.detectMultiScale(src(Range(140, 300), Range(0, 480)), found_tmp, weight, 0.2, hog3.blockStride, Size(0, 0), 1.05, 2);
	//0.2~0.25
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		if (r.tl().y <= 190 && r.br().y >= 230)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "large";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!�߽�ȷ��ע��(Range(300, 570), Range(0, 1280))

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��  
	int x1, x2, y1, y2;
	for (int i = 0; i < found.size(); i++)
	{
		mr = found[i];
		int j = 0;
		//for (; j < found.size(); j++)
		//	if (j != i && (r & found[j]) == r)
		//		break;
		for(; j <found.size(); j++)
		{	
			x1 = cvRound((mr.rect.tl().x + mr.rect.br().x) / 2);
			x2 = cvRound((found[j].rect.tl().x + found[j].rect.br().x) / 2);
			y1 = cvRound((mr.rect.tl().y + mr.rect.br().y) / 2);
			y2 = cvRound((found[j].rect.tl().y + found[j].rect.br().y) / 2);
			if (j != i)
				if ((mr.w < found[j].w))
					if(abs(x1 - x2) <= abs(cvRound(found[j].rect.width / 2)))
						if(abs(y1 - y2) <= abs(cvRound(found[j].rect.height / 2)))
							break;
			///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		}
		if (j == found.size())
			found_filtered.push_back(mr);
	}

	//found_filtered.insert(found_filtered.end(), found.begin(), found.end());

	string hePath;
	string headString;
	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����  
	//stringstream ss;
	for (int i = 0; i < found_filtered.size(); i++)
	{

		//int x1, y1, x2, y2;
		mr = found_filtered[i];
		//x1 = cvRound(((r.x + r.br().x) - winSize.width) / 2);
		//y1 = cvRound(((r.y + r.br().y) - winSize.height) / 2);
		//y1 += 170;
		//x2 = x1 + winSize.width;
		//y2 = y1 + winSize.height;
		//if (x1 < 0) {
		//	x1 = 0;
		//	x2 = winSize.width;
		//}
		//if (x2 > 480) {
		//	x1 = 480 - winSize.width;
		//	x2 = 480;
		//}
		//if (y2 > 356){
		//	y1 = 356 - winSize.height;
		//	y2 = 356;
		//}
	/*	if (TRAINTYPE == 1)
		{
			headString = "she_";
		}
		else if(TRAINTYPE == 2)
		{
			headString = "mhe_";
		}
		else if(TRAINTYPE == 3)
		{
			headString = "bhe_";
		}*/

		//ss.str("");
		//ss << i;
		//hePath = dirPath + headString + num + "_" + ss.str() + ".jpg";
		//imwrite(hePath, src(Range(y1, y2), Range(x1, x2)));
		
		mr.rect.x += cvRound(mr.rect.width*0.1);
		mr.rect.width = cvRound(mr.rect.width*0.8);
		mr.rect.y += 140;
		mr.rect.y += cvRound(mr.rect.height*0.07);
		mr.rect.height = cvRound(mr.rect.height*0.8);
		////!!!!������ı߽��Ӧ
		//rectangle(trtd, Rect(0, 120, 480, 180), Scalar(0, 255, 0), 1);//���½�
		//rectangle(trtd, Rect(0, 190, 480, 1), Scalar(255, 255, 255), 1);//��ƽ��
		//rectangle(trtd, Rect(0, 205, 480, 1), Scalar(255, 255, 0), 1);//30m��
		//rectangle(trtd, Rect(0, 220, 480, 1), Scalar(255, 0, 255), 1);//15m��
		//rectangle(trtd, Rect(0, 235, 480, 1), Scalar(0, 0, 255), 1);//10m��


		//if(r.tl().y <190 && r.br().y>190)
		rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(0, 255, 0), 1);
	}
	return;
}

void processedImgToVideo(string dirPath,char * videoPath,int tolFrame) {
	IplImage* img;
	string imgPath;
	char const *fimgPath;
	CvVideoWriter* writer = cvCreateVideoWriter(videoPath, CV_FOURCC('X', 'V', 'I', 'D'), 14, Size(480, 356));
	stringstream ss;
	for (int i = 0; i < tolFrame; i++)
	{
		ss.str("");
		ss << i;
		imgPath = dirPath + "pimage" + ss.str() + ".jpg";
		fimgPath = imgPath.c_str();
		img = cvLoadImage(fimgPath);
		cvWriteFrame(writer, img);
		cvReleaseImage(&img);
		cout << imgPath << endl;
	}
	cvReleaseVideoWriter(&writer);
}

//int main()
//{
//	bool bbbb = true;
//	if (bbbb == false)
//	{
//		int a = 1;
//		cout << a << endl;
//	}
//	stringstream ss;
//	int a = 100l;
//	int b = 2002;
//	ss << a;
//	cout << ss.str() << endl;
//	cout << "hhe" << endl;
//	ss.str("");
//	ss << b;
//	cout << ss.str() << endl;
//	system("pause");
	//cout<< CV_VERSION<<endl;

//
//	string detectDataPath = "D:\\detectProject\\data\\sourceData\\TRAINDATA\\pvideoList.txt";
//	string sourceDataPath = "D:\\detectProject\\data\\sourceData\\TRAINDATA\\videoList.txt";
//	ifstream finDetect(detectDataPath.data());
//	ifstream finSource(sourceDataPath.data());
//	int tolFrame;
//	string detectData, sourceData, dirPath,tmpVideoPath;
//	VideoCapture cap;
//	while (getline(finDetect, detectData))
//	{
//		getline(finSource, sourceData);
//		cap.open(sourceData.data());
//		if (!cap.isOpened()) {
//			cout<<"Cannot open the video."<<sourceData<<endl;
//			return -1;
//		}
//		tolFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
//
//		dirPath =detectData + "\\";
//		tmpVideoPath =detectData + "p.avi";
//		char* videoPath = _strdup(tmpVideoPath.c_str());
//		processedImgToVideo(dirPath, videoPath,tolFrame);
//		free(videoPath);
//	}
//}



int main_1()
{
	string posPath1, negPath1, hardPath1, detectorPath1, modelPath1, trainType, detectDataPath;
	string posPath2, negPath2, hardPath2, detectorPath2, modelPath2;
	string posPath3, negPath3, hardPath3, detectorPath3, modelPath3;
	int PosSamNO1, NegSamNO1, HardExampleNO1;
	int PosSamNO2, NegSamNO2, HardExampleNO2;
	int	PosSamNO3, NegSamNO3, HardExampleNO3;
	Size winSize1, blockSize1, blockStride1, cellSize1;
	Size winSize2, blockSize2, blockStride2, cellSize2;
	Size winSize3, blockSize3, blockStride3, cellSize3;
	Rect rectCrop1;
	Rect rectCrop2;
	Rect rectCrop3;
	//������������������������HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������  
	//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ  
	string configPath = ".\\ndsconfig.txt";
	ifstream configFile(configPath.data());
	getline(configFile, posPath1);
	cout << "Loading posPath_S:	"<< posPath1 << endl;
	getline(configFile, negPath1);
	cout << "Loading negPath_S:	" << negPath1 << endl;
	getline(configFile, hardPath1);
	cout << "Loading hardPath_S:	" << hardPath1 << endl;
	getline(configFile, detectorPath1);
	cout << "Loading detectorPath_S:	" << detectorPath1 << endl;
	getline(configFile, modelPath1);
	cout << "Loading modelPath_S:	" << modelPath1 << endl;
	getline(configFile, posPath2);
	cout << "Loading posPath_M:	" << posPath2 << endl;
	getline(configFile, negPath2);
	cout << "Loading negPath_M:	" << negPath2 << endl;
	getline(configFile, hardPath2);
	cout << "Loading hardPath_M:	" << hardPath2 << endl;
	getline(configFile, detectorPath2);
	cout << "Loading detectorPath_M:	" << detectorPath2 << endl;
	getline(configFile, modelPath2);
	cout << "Loading modelPath_M:	" << modelPath2 << endl;
	getline(configFile, posPath3);
	cout << "Loading posPath_L:	" << posPath3 << endl;
	getline(configFile, negPath3);
	cout << "Loading negPath_L:	" << negPath3 << endl;
	getline(configFile, hardPath3);
	cout << "Loading hardPath_L:	" << hardPath3<< endl;
	getline(configFile, detectorPath3);
	cout << "Loading detectorPath_L:	" << detectorPath3 << endl;
	getline(configFile, modelPath3);
	cout << "Loading modelPath_L:	" << modelPath3 << endl;
	getline(configFile, detectDataPath);
	cout << "Loading detectDataPath:	" << detectDataPath << endl;
	getline(configFile, trainType);
	PosSamNO1 = stoi(trainType);
	cout << "Loading posSamNum_S:	" << trainType << endl;
	getline(configFile, trainType);
	NegSamNO1 = stoi(trainType);
	cout << "Loading negSamNum_S:	" << trainType << endl;
	getline(configFile, trainType);
	HardExampleNO1 = stoi(trainType);
	cout << "Loading hardSamNum_S:	" << trainType << endl;
	getline(configFile, trainType);
	PosSamNO2 = stoi(trainType);
	cout << "Loading posSamNum_M:	" << trainType << endl;
	getline(configFile, trainType);
	NegSamNO2 = stoi(trainType);
	cout << "Loading negSamNum_M:	" << trainType << endl;
	getline(configFile, trainType);
	HardExampleNO2 = stoi(trainType);
	cout << "Loading hardSamNum_M:	" << trainType << endl;
	getline(configFile, trainType);
	PosSamNO3 = stoi(trainType);
	cout << "Loading posSamNum_L:	" << trainType << endl;
	getline(configFile, trainType);
	NegSamNO3 = stoi(trainType);
	cout << "Loading negSamNum_L:	" << trainType << endl;
	getline(configFile, trainType);
	HardExampleNO3 = stoi(trainType);
	cout << "Loading hardSamNum_L:	" << trainType << endl;
	getline(configFile, trainType);
	if (trainType == "1")
		TRAIN = true;
	cout << "Loading isTrain:	" << trainType << endl;
	getline(configFile, trainType);
	if (trainType == "1")
		CENTRAL_CROP = true;
	cout << "Loading isCrop:		" << trainType << endl;
	configFile.close();
	//getline(configFile, trainType);
	//TRAINTYPE = stoi(trainType);
	//cout << "Loading trainType:	" << trainType << endl;

	//if (1 == TRAINTYPE)
	//{
		//posPath = "D:\\detectProject\\SmallTrainData.txt";//������ͼƬ���ļ����б�
		//negPath = "D:\\detectProject\\NegativeData1.txt";//������ͼƬ���ļ����б�
		//hardPath = "";
		//modelPath = "D:\\detectProject\\model\\SVM_HOG_S.xml";
		//detectorPath = "D:\\detectProject\\model\\HOGDetector_S.txt";  
		
	winSize1 = Size(16, 32);
	blockSize1 = Size(4, 4);
	blockStride1 = Size(2, 2);
	cellSize1= Size(2, 2);
	rectCrop1 = Rect(0, 0, 16, 32);
		//winSize = Size(48, 96);
		//blockSize = Size(16, 16);
		//blockStride = Size(8, 8);
		//cellSize = Size(8, 8);
		//rectCrop = Rect(1, 2, 48, 96);
	//}
	//else if (2 == TRAINTYPE)
	//{
		//posPath = "D:\\detectProject\\MiddleTrainData.txt";//������ͼƬ���ļ����б�
		//negPath = "D:\\detectProject\\NegativeData2.txt";//������ͼƬ���ļ����б�
		//hardPath = "";
		//modelPath = "D:\\detectProject\\model\\SVM_HOG_M.xml";
		//detectorPath = "D:\\detectProject\\model\\HOGDetector_M.txt";
	winSize2 = Size(24, 48);
	blockSize2 = Size(8, 8);
	blockStride2 = Size(4, 4);
	cellSize2 = Size(4, 4);
	rectCrop2 = Rect(0, 1, 24, 48);
		//winSize = Size(96, 192);
		//blockSize = Size(16, 16);
		//blockStride = Size(8, 8);
		//cellSize = Size(8, 8);
		//rectCrop = Rect(2, 4, 96, 192);

	/*}
	else if (3 == TRAINTYPE)
	{*/
		//posPath = "D:\\detectProject\\LargeTrainData.txt";//������ͼƬ���ļ����б�
		//negPath = "D:\\detectProject\\NegativeData3.txt";//������ͼƬ���ļ����б�
		//hardPath = "";
		//modelPath = "D:\\detectProject\\model\\SVM_HOG_L.xml";
		//detectorPath = "D:\\detectProject\\model\\HOGDetector_L.txt";
	winSize3 = Size(48, 96);
	blockSize3 = Size(16, 16);
	blockStride3 = Size(8, 8);
	cellSize3 = Size(8, 8);
	rectCrop3 = Rect(1, 2, 48, 96);
		//winSize = Size(192, 384);
		//blockSize = Size(16, 16);
		//blockStride = Size(8, 8);
		//cellSize = Size(8, 8);
		//rectCrop = Rect(4, 8, 192, 384);
	//}

	HOGDescriptor hog1(winSize1, blockSize1, blockStride1, cellSize1, 9);
	HOGDescriptor hog2(winSize2, blockSize2, blockStride2, cellSize2, 9);
	HOGDescriptor hog3(winSize3, blockSize3, blockStride3, cellSize3, 9);
	vector<float> descriptors;
	if (TRAIN == true)
	{
		trainSVM(posPath1, negPath1, hardPath1, hog1, modelPath1, descriptors, PosSamNO1, NegSamNO1, HardExampleNO1);
		trainSVM(posPath2, negPath2, hardPath2, hog2, modelPath2, descriptors, PosSamNO2, NegSamNO2, HardExampleNO2);
		trainSVM(posPath3, negPath3, hardPath3, hog3, modelPath3, descriptors, PosSamNO3, NegSamNO3, HardExampleNO3);
	}
	MySVM svm1, svm2, svm3;
	vector<float> myDetector;
	svm1.load(modelPath1.data());
	setDetector(svm1, myDetector, detectorPath1);
	hog1.setSVMDetector(myDetector);
	myDetector.clear();
	svm2.load(modelPath2.data());
	setDetector(svm2, myDetector, detectorPath2);
	hog2.setSVMDetector(myDetector);
	myDetector.clear();
	svm3.load(modelPath3.data());
	setDetector(svm3, myDetector, detectorPath3);
	hog3.setSVMDetector(myDetector);
	myDetector.clear();

	/**************����ͼƬ����HOG���˼��******************/
	cout << "Start Detecting..." << endl;
	vector<Rect> found_tmp;//���ο�����
	vector<myRect> found_filtered, found;
	vector<double> weight;
	ifstream finDetect(detectDataPath.data());
	string detectData, videoPath, rectFilePath;
	Mat src,trtd;
	IplImage* iplimage;
	string imgPath;
	stringstream ss;
	VideoCapture cap;
	CvVideoWriter* writer;
	double totalFrame;
	
	while(getline(finDetect, detectData))
	{
		cout << "Detecting "<<detectData << endl;
		videoPath = detectData;
		cap.open(videoPath.data());
		if (!cap.isOpened()) {
			cout<<"Cannot open the video."<<videoPath<<endl;
			continue;
		}
		totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
		videoPath = videoPath.substr(0, detectData.length() - 4) + "p.avi";
		rectFilePath = videoPath.substr(0, detectData.length() - 4) + "r.txt";
		ofstream fout(rectFilePath.data());
		writer = cvCreateVideoWriter(videoPath.data(), CV_FOURCC('X', 'V', 'I', 'D'), 14, Size(480, 356));
		for(int num = 0;num<totalFrame;num++){
			ss.str("");
			ss << num;
			cap.read(src);
			trtd = src.clone();
			DetectAndDraw(src, trtd, hog1, hog2, hog3, found, found_tmp, found_filtered, weight);
			//detectData.substr(0, detectData.length() - 4) +"_"+ss.str()
			/*if (_access((detectData.substr(0, detectData.length() -4)).data(), 0) == -1) {
				_mkdir((detectData.substr(0, detectData.length() - 4)).data());
				cout << detectData.substr(0, detectData.length() - 4) << endl;
			}*/

			iplimage = &IplImage(trtd);
			cvWriteFrame(writer, iplimage);
//			cvReleaseImage(&iplimage);

			//imgPath = detectData.substr(0, detectData.length() - 4) + "\\pimage" + ss.str() + ".jpg";
			for (int i = 0; i < found_filtered.size(); i++)
			{
				fout << found_filtered[i].rect.tl().x << " " << found_filtered[i].rect.tl().y << " "
					<< found_filtered[i].rect.br().x << " " << found_filtered[i].rect.br().y << " "
					<< found_filtered[i].group << ",";
			}
			fout << endl;
			found.clear();
			found_tmp.clear();
			weight.clear();
			found_filtered.clear();
			//imwrite(imgPath, trtd);
		}
		fout.close();
		cvReleaseVideoWriter(&writer);
		cap.release();
	}
	finDetect.close();

//	namedWindow("src", 0);
//	imshow("src", trtd);
//	waitKey();//ע�⣺imshow֮������waitKey�������޷���ʾͼ��  
	system("pause");
	return 0;
}

//���ζ�ȡ������ͼƬ������HOG������  
//for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
//{
//	//cout << "����" << ImgName << num << endl;
//	ImgName = "D:\\detectProject\\traindata\\" + ImgName;//������������·����  
//	Mat src = imread(ImgName);//��ȡͼƬ  
//	//imshow("....", src);
//	//waitKey(6000);
//	if (CENTRAL_CROP)
//		src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������  
//										 //resize(src,src,Size(64,128));  
//	hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//											  //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������  
//	cout << descriptors.size() << endl;
//	if (0 == num)
//	{
//		DescriptorDim = descriptors.size();//HOG�����ӵ�ά��  
//										   //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat  
//		sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
//		//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����  
//		sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
//	}
//	//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
//	sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
//	descriptors.clear();
//}

////���ζ�ȡ������ͼƬ������HOG������  
//for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
//{
//	//cout << "����" << ImgName << num << endl;
//	ImgName = "D:\\detectProject\\negativedata\\" + ImgName;//���ϸ�������·����  
//	Mat src = imread(ImgName);//��ȡͼƬ  
//							  //resize(src,img,Size(64,128));  
//	//imshow("....", src);
//	//waitKey(6000);
//	hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//											  //cout<<"������ά����"<<descriptors.size()<<endl;  
//											  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��  
//	sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1������  
//	descriptors.clear();
//}

//for (int num = 0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
//{
//	cout << "����" << ImgName << endl;
//	ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����  
//	Mat src = imread(ImgName);//��ȡͼƬ  
//							  //resize(src,img,Size(64,128));  
//	hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//											  //cout<<"������ά����"<<descriptors.size()<<endl; 
//											  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��  
//	sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������  
//	descriptors.clear();
//}


////��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9  
//HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);//HOG���������������HOG�����ӵ�  
//int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������  
//MySVM svm;//SVM������
//vector<float> descriptors;//HOG����������
////namedWindow("~.~");
//		  //��TRAINΪtrue������ѵ��������  
//if (TRAIN)
//{
//	string ImgName;//ͼƬ��(����·��)  
//	ifstream finPos("D:\\detectProject\\LargeTrainData.txt");//������ͼƬ���ļ����б�  
//	ifstream finNeg("D:\\detectProject\\NegativeData3.txt");//������ͼƬ���ļ����б�  

//	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��      
//	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����  

//	string trainPath = "D:\\detectProject\\traindata\\";
//	string bgPath = "D:\\detectProject\\negativedata\\";
//	//���ζ�ȡ������ͼƬ������HOG������  
//	generateDescriptors(finPos, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 0, trainPath);
//	//���ζ�ȡ������ͼƬ������HOG������  
//	generateDescriptors(finNeg, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 1, bgPath);
//	
//	//����HardExample������  
//	if (HardExampleNO > 0)
//	{
//		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample������ͼƬ���ļ����б�
//		string hardPath = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\";
//		generateDescriptors(finHardExample, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 2, hardPath);																	 //���ζ�ȡHardExample������ͼƬ������HOG������  
//	}

//	////���������HOG�������������ļ�  
//	/*ofstream fout("D:\\detectProject\\SampleFeatureMat.txt");  
//	for(int i=0; i<PosSamNO+NegSamNO; i++)  
//	{  
//	  fout<<i<<endl;  
//	  for(int j=0; j<DescriptorDim; j++)  
//	      fout<<sampleFeatureMat.at<float>(i,j)<<"  ";  
//	  fout<<endl;  
//	} */ 

//	//ѵ��SVM������  
//	//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����  
//	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
//	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01  
//	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
//	cout << "��ʼѵ��SVM������" << endl;
//	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������  
//	cout << "ѵ�����" << endl;
//	svm.save("D:\\detectProject\\model\\SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  
//}
//else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����  
//{
//	svm.load("D:\\detectProject\\model\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��  
//}


//int DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��  
//int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���  
////cout << "֧������������" << supportVectorNum << endl;

//Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������  
//Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������  
//Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��  

//													   //��֧�����������ݸ��Ƶ�supportVectorMat������  
//for (int i = 0; i < supportVectorNum; i++)
//{
//	const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��  
//	for (int j = 0; j < DescriptorDim; j++)
//	{
//		//cout<<pData[j]<<" ";  
//		supportVectorMat.at<float>(i, j) = pSVData[j];
//	}
//}

////��alpha���������ݸ��Ƶ�alphaMat��  
//double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
//for (int i = 0; i < supportVectorNum; i++)
//{
//	alphaMat.at<float>(0, i) = pAlphaData[i];
//}

////����-(alphaMat * supportVectorMat),����ŵ�resultMat��  
////gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�  
//resultMat = -1 * alphaMat * supportVectorMat;

////�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����  
//vector<float> myDetector;
////��resultMat�е����ݸ��Ƶ�����myDetector��  
//for (int i = 0; i < DescriptorDim; i++)
//{
//	myDetector.push_back(resultMat.at<float>(0, i));
//}
////������ƫ����rho���õ������  
//myDetector.push_back(svm.get_rho());
//cout << "�����ά����" << myDetector.size() << endl;
////����HOGDescriptor�ļ����  
//hog.setSVMDetector(myDetector);
////myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());  

////�������Ӳ������ļ�  
//ofstream fout("D:\\detectProject\\HOGDetectorForOpenCV.txt");
//for (int i = 0; i < myDetector.size(); i++)
//{
//	fout << myDetector[i] << endl;
//}

/******************���뵥��64*128�Ĳ���ͼ������HOG�����ӽ��з���*********************/
////��ȡ����ͼƬ(64*128��С)����������HOG������  
////Mat testImg = imread("person014142.jpg");  
//Mat testImg = imread("noperson000026.jpg");  
//vector<float> descriptor;  
//hog.compute(testImg,descriptor,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//����������������������  
////������õ�HOG�����Ӹ��Ƶ�testFeatureMat������  
//for(int i=0; i<descriptor.size(); i++)  
//  testFeatureMat.at<float>(0,i) = descriptor[i];  

////��ѵ���õ�SVM�������Բ���ͼƬ�������������з���  
//int result = svm.predict(testFeatureMat);//�������  
//cout<<"��������"<<result<<endl;  

////cout << "���ж�߶�HOG������" << endl;
//hog.detectMultiScale(src(Range(300, 720), Range(0, 1280)), found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//��ͼƬ���ж�߶����˼��  
////!!!!!!!!!!!!!!!!!!!!!!!!!!!�߽�ȷ��ע��
////cout << "�ҵ��ľ��ο������" << found.size() << endl;

////�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��  
//for (int i = 0; i < found.size(); i++)
//{
//	Rect r = found[i];
//	int j = 0;
//	for (; j < found.size(); j++)
//		if (j != i && (r & found[j]) == r)
//			break;
//	if (j == found.size())
//		found_filtered.push_back(r);
//}
////�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����  
//for (int i = 0; i < found_filtered.size(); i++)
//{
//	Rect r = found_filtered[i];
//	r.x += cvRound(r.width*0.1);
//	r.width = cvRound(r.width*0.8);
//	r.y += cvRound(r.height*0.07);
//	r.y += 300;
//	//!!!!������ı߽��Ӧ
//	r.height = cvRound(r.height*0.8);
//	rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
//}