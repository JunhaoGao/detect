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

bool TRAIN = false;   //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型  
bool CENTRAL_CROP = false;   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体  
//int TRAINTYPE = 0;


//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，  
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问  
class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组  
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量  
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
		Mat src = imread(imgName);//读取图片  

		if (CENTRAL_CROP)
			resize(src, src, hog.winSize);
			//src = src(rectCrop);//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素  
								/*		imshow("....", src);
								waitKey(6000);			*/							 //resize(src,src,Size(64,128));  
		hog.compute(src, descriptors, hog.blockStride);//计算HOG描述子，检测窗口移动步长(8,8)  
												  //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵  
												  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat  
		if (0 == trainClass)
		{
			if (0 == num)
			{
				descriptorDim = descriptors.size();	//HOG描述子的维数 
													//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat  
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, descriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人  
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}
		else if (1 == trainClass) {
			if (0 == num)
				descriptorDim = sampleFeatureMat.cols;
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//正样本类别为1，有人
		}
		else if (2 == trainClass)
		{
			if (0 == num)
				descriptorDim = sampleFeatureMat.cols;
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
			sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//正样本类别为1，有人
		}

	}
	descriptors.clear();
	return;
}

void trainSVM(string posPath,string negPath, string hardPath, HOGDescriptor& hog, string modelPath, vector<float>& descriptors, int PosSamNO, int NegSamNO, int HardExampleNO) {

	ifstream finPos(posPath.data());
	ifstream finNeg(negPath.data());
	ifstream finHard(hardPath.data());
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定  
	MySVM svm;//SVM分类器
	//HOG描述子向量
	string ImgName;//图片名(绝对路径) 
	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数      
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人  

	cout << "开始计算正样本检测子" << endl;
	generateDescriptors(finPos, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 0, PosSamNO, NegSamNO, HardExampleNO);
	cout << "计算完成" << endl;
	cout << "开始计算负样本检测子" << endl;
	generateDescriptors(finNeg, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 1, PosSamNO, NegSamNO, HardExampleNO);
	cout << "计算完成" << endl;
	if (HardExampleNO > 0)
		//依次读取HardExample负样本图片，生成HOG描述子  
		generateDescriptors(finHard, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 2, PosSamNO, NegSamNO, HardExampleNO);
	
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01  
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	cout << "开始训练SVM分类器" << endl;
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器  
	cout << "训练完成" << endl;
	svm.save(modelPath.data());//将训练好的SVM模型保存为xml文件 
	descriptors.clear();
	finPos.close();
	finNeg.close();
	finHard.close();
	return;
}
	/*******************************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()）
	，就可以利用你的训练样本训练出来的分类器进行行人检测了。
	********************************************************************************************************************/
void setDetector(MySVM& svm, vector<float>& myDetector, string detectorPath){
	int DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数  
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数  
														  //cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数  
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵  
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果  

														   //将支持向量的数据复制到supportVectorMat矩阵中  
	for (int i = 0; i < supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针  
		for (int j = 0; j < DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";  
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中  
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中  
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？  
	resultMat = -1 * alphaMat * supportVectorMat;

	//将resultMat中的数据复制到数组myDetector中  
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子  
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;

	//保存检测子参数到文件  
	ofstream fout(detectorPath.data());
	for (int i = 0; i < myDetector.size(); i++)
		fout << myDetector[i] << endl;
	fout.close();
	
	return;
}

void DetectAndDraw(Mat& src, Mat &trtd, HOGDescriptor& hog1, HOGDescriptor& hog2, HOGDescriptor& hog3, vector<myRect>& found,
	vector<Rect>& found_tmp, vector<myRect>& found_filtered, vector<double>& weight){
	//string path
	//对图片进行多尺度行人检测
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
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!边界确定注意(Range(300, 570), Range(0, 1280))

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中  
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
	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整  
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
		////!!!!与上面的边界对应
		//rectangle(trtd, Rect(0, 120, 480, 180), Scalar(0, 255, 0), 1);//上下界
		//rectangle(trtd, Rect(0, 190, 480, 1), Scalar(255, 255, 255), 1);//视平线
		//rectangle(trtd, Rect(0, 205, 480, 1), Scalar(255, 255, 0), 1);//30m线
		//rectangle(trtd, Rect(0, 220, 480, 1), Scalar(255, 0, 255), 1);//15m线
		//rectangle(trtd, Rect(0, 235, 480, 1), Scalar(0, 0, 255), 1);//10m线


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
	//正样本个数，负样本个数，HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。  
	//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值  
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
		//posPath = "D:\\detectProject\\SmallTrainData.txt";//正样本图片的文件名列表
		//negPath = "D:\\detectProject\\NegativeData1.txt";//负样本图片的文件名列表
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
		//posPath = "D:\\detectProject\\MiddleTrainData.txt";//正样本图片的文件名列表
		//negPath = "D:\\detectProject\\NegativeData2.txt";//负样本图片的文件名列表
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
		//posPath = "D:\\detectProject\\LargeTrainData.txt";//正样本图片的文件名列表
		//negPath = "D:\\detectProject\\NegativeData3.txt";//负样本图片的文件名列表
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

	/**************读入图片进行HOG行人检测******************/
	cout << "Start Detecting..." << endl;
	vector<Rect> found_tmp;//矩形框数组
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
//	waitKey();//注意：imshow之后必须加waitKey，否则无法显示图像  
	system("pause");
	return 0;
}

//依次读取正样本图片，生成HOG描述子  
//for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
//{
//	//cout << "处理：" << ImgName << num << endl;
//	ImgName = "D:\\detectProject\\traindata\\" + ImgName;//加上正样本的路径名  
//	Mat src = imread(ImgName);//读取图片  
//	//imshow("....", src);
//	//waitKey(6000);
//	if (CENTRAL_CROP)
//		src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素  
//										 //resize(src,src,Size(64,128));  
//	hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)  
//											  //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵  
//	cout << descriptors.size() << endl;
//	if (0 == num)
//	{
//		DescriptorDim = descriptors.size();//HOG描述子的维数  
//										   //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat  
//		sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
//		//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人  
//		sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
//	}
//	//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
//	sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
//	descriptors.clear();
//}

////依次读取负样本图片，生成HOG描述子  
//for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
//{
//	//cout << "处理：" << ImgName << num << endl;
//	ImgName = "D:\\detectProject\\negativedata\\" + ImgName;//加上负样本的路径名  
//	Mat src = imread(ImgName);//读取图片  
//							  //resize(src,img,Size(64,128));  
//	//imshow("....", src);
//	//waitKey(6000);
//	hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)  
//											  //cout<<"描述子维数："<<descriptors.size()<<endl;  
//											  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素  
//	sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//负样本类别为-1，无人  
//	descriptors.clear();
//}

//for (int num = 0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
//{
//	cout << "处理：" << ImgName << endl;
//	ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名  
//	Mat src = imread(ImgName);//读取图片  
//							  //resize(src,img,Size(64,128));  
//	hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)  
//											  //cout<<"描述子维数："<<descriptors.size()<<endl; 
//											  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素  
//	sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人  
//	descriptors.clear();
//}


////检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9  
//HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);//HOG检测器，用来计算HOG描述子的  
//int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定  
//MySVM svm;//SVM分类器
//vector<float> descriptors;//HOG描述子向量
////namedWindow("~.~");
//		  //若TRAIN为true，重新训练分类器  
//if (TRAIN)
//{
//	string ImgName;//图片名(绝对路径)  
//	ifstream finPos("D:\\detectProject\\LargeTrainData.txt");//正样本图片的文件名列表  
//	ifstream finNeg("D:\\detectProject\\NegativeData3.txt");//负样本图片的文件名列表  

//	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数      
//	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人  

//	string trainPath = "D:\\detectProject\\traindata\\";
//	string bgPath = "D:\\detectProject\\negativedata\\";
//	//依次读取正样本图片，生成HOG描述子  
//	generateDescriptors(finPos, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 0, trainPath);
//	//依次读取负样本图片，生成HOG描述子  
//	generateDescriptors(finNeg, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 1, bgPath);
//	
//	//处理HardExample负样本  
//	if (HardExampleNO > 0)
//	{
//		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample负样本图片的文件名列表
//		string hardPath = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\";
//		generateDescriptors(finHardExample, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 2, hardPath);																	 //依次读取HardExample负样本图片，生成HOG描述子  
//	}

//	////输出样本的HOG特征向量矩阵到文件  
//	/*ofstream fout("D:\\detectProject\\SampleFeatureMat.txt");  
//	for(int i=0; i<PosSamNO+NegSamNO; i++)  
//	{  
//	  fout<<i<<endl;  
//	  for(int j=0; j<DescriptorDim; j++)  
//	      fout<<sampleFeatureMat.at<float>(i,j)<<"  ";  
//	  fout<<endl;  
//	} */ 

//	//训练SVM分类器  
//	//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代  
//	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
//	//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01  
//	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
//	cout << "开始训练SVM分类器" << endl;
//	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器  
//	cout << "训练完成" << endl;
//	svm.save("D:\\detectProject\\model\\SVM_HOG.xml");//将训练好的SVM模型保存为xml文件  
//}
//else //若TRAIN为false，从XML文件读取训练好的分类器  
//{
//	svm.load("D:\\detectProject\\model\\SVM_HOG.xml");//从XML文件读取训练好的SVM模型  
//}


//int DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数  
//int supportVectorNum = svm.get_support_vector_count();//支持向量的个数  
////cout << "支持向量个数：" << supportVectorNum << endl;

//Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数  
//Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵  
//Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果  

//													   //将支持向量的数据复制到supportVectorMat矩阵中  
//for (int i = 0; i < supportVectorNum; i++)
//{
//	const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针  
//	for (int j = 0; j < DescriptorDim; j++)
//	{
//		//cout<<pData[j]<<" ";  
//		supportVectorMat.at<float>(i, j) = pSVData[j];
//	}
//}

////将alpha向量的数据复制到alphaMat中  
//double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
//for (int i = 0; i < supportVectorNum; i++)
//{
//	alphaMat.at<float>(0, i) = pAlphaData[i];
//}

////计算-(alphaMat * supportVectorMat),结果放到resultMat中  
////gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？  
//resultMat = -1 * alphaMat * supportVectorMat;

////得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子  
//vector<float> myDetector;
////将resultMat中的数据复制到数组myDetector中  
//for (int i = 0; i < DescriptorDim; i++)
//{
//	myDetector.push_back(resultMat.at<float>(0, i));
//}
////最后添加偏移量rho，得到检测子  
//myDetector.push_back(svm.get_rho());
//cout << "检测子维数：" << myDetector.size() << endl;
////设置HOGDescriptor的检测子  
//hog.setSVMDetector(myDetector);
////myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());  

////保存检测子参数到文件  
//ofstream fout("D:\\detectProject\\HOGDetectorForOpenCV.txt");
//for (int i = 0; i < myDetector.size(); i++)
//{
//	fout << myDetector[i] << endl;
//}

/******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
////读取测试图片(64*128大小)，并计算其HOG描述子  
////Mat testImg = imread("person014142.jpg");  
//Mat testImg = imread("noperson000026.jpg");  
//vector<float> descriptor;  
//hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)  
//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵  
////将计算好的HOG描述子复制到testFeatureMat矩阵中  
//for(int i=0; i<descriptor.size(); i++)  
//  testFeatureMat.at<float>(0,i) = descriptor[i];  

////用训练好的SVM分类器对测试图片的特征向量进行分类  
//int result = svm.predict(testFeatureMat);//返回类标  
//cout<<"分类结果："<<result<<endl;  

////cout << "进行多尺度HOG人体检测" << endl;
//hog.detectMultiScale(src(Range(300, 720), Range(0, 1280)), found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//对图片进行多尺度行人检测  
////!!!!!!!!!!!!!!!!!!!!!!!!!!!边界确定注意
////cout << "找到的矩形框个数：" << found.size() << endl;

////找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中  
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
////画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整  
//for (int i = 0; i < found_filtered.size(); i++)
//{
//	Rect r = found_filtered[i];
//	r.x += cvRound(r.width*0.1);
//	r.width = cvRound(r.width*0.8);
//	r.y += cvRound(r.height*0.07);
//	r.y += 300;
//	//!!!!与上面的边界对应
//	r.height = cvRound(r.height*0.8);
//	rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
//}