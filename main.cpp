#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define N_INPUT 307200
#define N_HIDDEN 45
#define N_OUTPUT 15
#define ITERATIONS 6000
#define ERROR 0.01

String path_dir = "/home/antonio/Documentos/SensoresInteligentes/Base/test/"

void Treina_MLP(Mat trainingDataMat, Mat labelsMat) { 
    Mat net = Mat(1,3,CV_32SC1);
    net.at<int>(0,0) = N_INPUT;
    net.at<int>(0,1) = N_HIDDEN;
    net.at<int>(0,2) = N_OUTPUT;
    CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, ITERATIONS, ERROR),
                                                         CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
    
    CvANN_MLP *mlp = new CvANN_MLP();
    mlp->create(net, CvANN_MLP::SIGMOID_SYM, 0.6, 1);
    mlp->train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    mlp->save("mlp.xml");
}

void Testa_MLP(Mat trainingDataMat, Mat labelsMat) {
    CvANN_MLP *mlp = new CvANN_MLP();
    mlp->load("mlp.xml");
    Mat resp(N_OUTPUT, N_OUTPUT, CV_32FC1);
    mlp->predict(trainingDataMat, resp);
    
    int errados = 0, total = 0;
    int predict = 0;
    for (int i = 0; i < trainingDataMat.rows; i++) {
        cout << " Predict: " << resp.at<float>(i) << " Erro: " << labelsMat.at<float>(i) - resp.at<float>(i) << endl;
        
        predict = (int)(resp.at<float>(i) + 0.5);
        total++;
        if(predict != (int)(labelsMat.at<float>(i)))  
            errados++;
    }
    cout << "Total: " << total << " Acuracia: " << 100.0 * ((float) (total - errados) / (float) total) << "%" << endl;
}

int main(int argc, char **argv) {
    Mat image;
    
    if (flag == 1) {
        Mat labelsMat = Mat::zeros(N_OUTPUT, N_OUTPUT, CV_32FC1);
        Mat trainingDataMat = Mat::zeros(N_OUTPUT, N_INPUT, CV_32FC1); 
        
        for (int k = 0; k < N_OUTPUT; k++) {
            stringstream ss;
            ss << k;
            String image_file = ss.str() + ".jpg";
            image = imread(path_dir + image_file, CV_LOAD_IMAGE_GRAYSCALE);
            for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                    trainingDataMat[k][i*image.cols+j] = image[i][j]/255.0;
                }
            }
            labelsMat[k][k] = 1;
        }
        
        Treina_MLP(trainingDataMat, labelsMat);
    }
    else if (flag == 2) {
        Mat labelsMat = Mat::zeros(1, N_OUTPUT, CV_32FC1);
        Mat trainingDataMat = Mat::zeros(1, N_INPUT, CV_32FC1);
        int index = 0;
        String image_file;
        
        cout << "Digite o nome da imagem:" << endl;
        cin >> image_file;
        cin.ignore();
        cout << "Digite o indice da saida esperada: " << endl;
        cout << "[Indice da Saida]" << endl << "(0 1 2 3 4)" << endl << "(5 6 7 8 9)" << endl << "(10 11 12 13 14)" << endl;
        cin >> index;
        cin.ignore();
        image = imread(path_dir + image_file, CV_LOAD_IMAGE_GRAYSCALE);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                trainingDataMat[0][i*image.cols+j] = image[i][j]/255.0;
            }
        }
        
        labelsMat[0][index] = 1;
        Testa_MLP(trainingDataMat, labelsMat);
    }
    else if (flag == 3) {
        Mat trainingDataMat = Mat::zeros(1, N_INPUT, CV_32FC1);
        Mat labelsMat = Mat::zeros(1,N_OUTPUT,CV_32FC1);
        Mat lastResult = Mat::zeros(1,N_OUTPUT,CV_32FC1);
        Mat actualResult = Mat::zeros(1,N_OUTPUT,CV_32FC1);
        VideoCapture cap(0);
        if(!cap.isOpened())
            return -1;
        
        while(1) {
            Mat frame;
            cap >> frame;
            imshow("img2",frame);
            
            Mat gray = cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (int i = 0; i < (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT); i++) {
                for (int j = 0; j < (int) cap.get(CV_CAP_PROP_FRAME_WIDTH); j++) {
                    trainingDataMat[0][(i*((int)cap.get(CV_CAP_PROP_FRAME_WIDTH)))+j] = frame[i][j]/255.0;   
                }
            }
            
            actualResult = Testa_MLP(trainingDataMat,labelsMat);
            CompareResult(actualResult,lastResult);
            lastResult = actualResult;
            
            if (waitkey(30) >= 0)
                break;
        }
    }
    
    return 0;
}
