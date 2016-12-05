#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

#define N_INPUT 307200  // numero de neuronios na input layer
#define N_HIDDEN 100     // numero de neuronios na hidden layer
#define N_OUTPUT 15     // numero de neuronios na output layer
#define ITERATIONS 6000 // numero maximo de iteracoes
#define ERROR 0.0001  // erro minimo - threshold

String path_dir = "/home/antonio/Documentos/SensoresInteligentes/Base/test/";

/* Funcao que treina a Multilayer Perceptron
 * Parametros:
 *              trainingDataMat - base de dados do treinamento
 *              labelsMat       - saida esperada de cada dado de treinamento
 */
void Treina_MLP(Mat trainingDataMat, Mat labelsMat) { 
    Mat net = Mat::zeros(1,3,CV_32SC1); // cria os dados de cada camada de neuronios
    net.at<int>(0,0) = N_INPUT;
    net.at<int>(0,1) = N_HIDDEN;
    net.at<int>(0,2) = N_OUTPUT;
    // carrega os parametros para o backpropagation
    TermCriteria params = TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, ITERATIONS, ERROR);
    
    // cria a MLP com os dados da funcao de ativacao e os parametros definidos anteriormente, salvando apos o treino
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
    mlp->setLayerSizes(net);
    mlp->setActivationFunction(cv::ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM, 1, 1);
    mlp->setTermCriteria(params);
    mlp->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP, 0.0001, 0.0001);
    mlp->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
    mlp->save("mlp.xml");
}

/* Funcao que testa a Multilayer Perceptron previamente treinada
 * Parametros:
 *              trainingDataMat - base de dados de teste
 *              labelsMat       - saida esperada do teste
 *              index           - indice da saida esperada
 * Retorno:
 *              index   - caso a taxa de acerto seja maior que 0.5
 *              -1      - caso contrario
 */
int Testa_MLP(Mat trainingDataMat, Mat labelsMat, int index) {
    // carrega os dados treinados e testa com a base inserida
    Ptr<cv::ml::ANN_MLP> mlp = Algorithm::load<cv::ml::ANN_MLP>("mlp.xml");
    Mat resp(1, N_OUTPUT, CV_32FC1);
    mlp->predict(trainingDataMat, resp);
    
    // para treinamento em tempo real, seleciona o indice com menor erro
    //if (index < 0) {
        float min = 1.0;
        for (int i = 0; i < N_OUTPUT; i++) {
            float erro = resp.at<float>(0,i) - labelsMat.at<float>(0,i);
            cout << "Predict: " << resp.at<float>(0,i) << " Erro: " << resp.at<float>(0,i) - labelsMat.at<float>(0,i) << endl;
            if((erro >= 0) && (erro < min)) {
                min = erro;
                index = i;
            }
        }
    //}
    
    // calcula a taxa de acerto
    int errados = 0;
    int predict = 0;
    cout << "Predict: " << resp.at<float>(0,index) << " Erro: " << resp.at<float>(0,index) - labelsMat.at<float>(0,index) << endl;
    predict = (int)(resp.at<float>(0,index) + 0.5);
    if(predict != (int)(labelsMat.at<float>(0,index)))
        errados++;
    cout << "Erros: " << errados << endl;
    if(errados)
        return -1;
    
    return index;
}

/* Funcao que imprime qual movimento foi feito
 * Parametros:
 *              actualResult    - posicao da mao atual
 *              lastResult      - posicao da mao anterior
 * Referencia da posicao na tela:
 * 0    1   2   3   4
 * 5    6   7   8   9
 * 10   11  12  13  14
 */
void CompareResult(int actualResult, int lastResult) {
    if (lastResult < 0 || actualResult < 0)
        return;
    
    String movimento;
    
    if ((lastResult - actualResult == 5) || (lastResult - actualResult == 10))
        movimento = "cima";
    else if ((actualResult - lastResult == 5) || (actualResult - lastResult == 10))
        movimento = "baixo";
    else if ((actualResult - lastResult < 5) && (actualResult - lastResult >= 0)) {
        if ((actualResult%5) > (lastResult%5))
            movimento = "direita";
        else if ((actualResult%5) < (lastResult%5))
            movimento = "diagonal esquerda para baixo";
    }
    else if ((lastResult - actualResult < 5) && (lastResult - actualResult >= 0)) {
        if ((actualResult%5) < (lastResult%5))
            movimento = "esquerda";
        else if ((actualResult%5) > (lastResult%5))
            movimento = "diagonal direita para cima";
    }
    else if ((actualResult - lastResult > 5) && (actualResult - lastResult <= 14)) {
        if ((actualResult%5) < (lastResult%5))
            movimento = "diagonal esquerda para baixo";
        else if ((actualResult%5) > (lastResult%5))
            movimento = "diagonal direita para baixo";
    }
    else if ((lastResult - actualResult > 5) && (lastResult - actualResult <= 14)) {
        if ((actualResult%5) < (lastResult%5))
            movimento = "diagonal esquerda para cima";
        else if ((actualResult%5) > (lastResult%5))
            movimento = "diagonal direita para cima";
    }
    
    cout << movimento << endl;
}

int main(int argc, char **argv) {
    Mat image;
    int flag = 0;
    
    cout << "Escolha um opcao:" << endl << "[1] Treina a Rede" << endl << "[2] Classifica uma imagem" << endl 
        << "[3] Analisa sequencia de movimentos" << endl;
    cin >> flag;
    cin.ignore();
    
    if (flag == 1) {
        Mat labelsMat = Mat::zeros(N_OUTPUT, N_OUTPUT, CV_32FC1);
        Mat trainingDataMat = Mat::zeros(N_OUTPUT, N_INPUT, CV_32FC1); 
        
        for (int k = 0; k < N_OUTPUT; k++) {
            stringstream ss;
            ss << k;
            String image_file = ss.str() + ".jpg";
            image = imread(path_dir + image_file, CV_LOAD_IMAGE_GRAYSCALE);
            Mat aux;
            image.convertTo(aux, CV_32FC1);
            for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                    trainingDataMat.at<float>(k,i*image.cols+j) = aux.at<float>(i,j)/255.0;
                }
            }
            
            labelsMat.at<float>(k,k) = 1;
        }
        
        Treina_MLP(trainingDataMat, labelsMat);
    }
    else if (flag == 2) {
        Mat labelsMat = Mat::zeros(1, N_OUTPUT, CV_32FC1);
        Mat trainingDataMat = Mat::zeros(1, N_INPUT, CV_32FC1);
        int index = 0;
        string image_file;
        
        cout << "Digite o nome da imagem:" << endl;
        getline (cin, image_file);
        cout << "Digite o indice da saida esperada: " << endl;
        cout << "[Indice da Saida]" << endl << "(0 1 2 3 4)" << endl << "(5 6 7 8 9)" << endl << "(10 11 12 13 14)" << endl;
        cin >> index;
        cin.ignore();
        image = imread(path_dir + image_file, CV_LOAD_IMAGE_GRAYSCALE);
        Mat aux;
        image.convertTo(aux, CV_32FC1);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                trainingDataMat.at<float>(0,i*image.cols+j) = aux.at<float>(i,j)/255.0;
            }
        }
        
        labelsMat.at<float>(0,index) = 1;
        Testa_MLP(trainingDataMat, labelsMat,index);
    }
    else if (flag == 3) {
        Mat trainingDataMat = Mat::zeros(1, N_INPUT, CV_32FC1);
        Mat labelsMat = Mat::zeros(1,N_OUTPUT,CV_32FC1);
        int lastResult = -1, actualResult = -1;
        VideoCapture cap(0);
        if(!cap.isOpened())
            return -1;
        
        while(1) {
            Mat frame;
            cap >> frame;
            imshow("img2",frame);
            
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            for (int i = 0; i < (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT); i++) {
                for (int j = 0; j < (int) cap.get(CV_CAP_PROP_FRAME_WIDTH); j++) {
                    trainingDataMat.at<float>(0,(i*((int)cap.get(CV_CAP_PROP_FRAME_WIDTH)))+j) = frame.at<float>(i,j)/255.0;   
                }
            }
            
            actualResult = Testa_MLP(trainingDataMat,labelsMat,-1);
            CompareResult(actualResult,lastResult);
            lastResult = actualResult;
            
            if(waitKey(30) >= 0)
                break;
        }
    }
    
    return 0;
}
