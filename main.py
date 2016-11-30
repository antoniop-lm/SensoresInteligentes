from scipy import misc
import random
import numpy as np
import cv2
import pickle

N_HIDDEN = 45
N_OUTPUT = 15
ITERATIONS = 6000
ERROR = 0.01

# Classe com as imagens abertas
# image -> matriz com os valores dos pixels da imagem
# size -> tamanho total da imagem
# realOutput -> saida esperada
class Images:
    def __init__(self,image,index):
        self.image = image
        self.size = len(image) * len(image[0])
        self.realOutput = [0] * 15
        self.realOutput[index] = 1
        
    def setImageIndex(self,i,j,value):
        self.image[i][j] = value
        
    def getSize(self):
        return self.size
        
    def getImage(self):
        return self.image
    
    def getImageIndex(self,i,j):
        return self.image[i][j]
    
    def getImageWidth(self):
        return len(self.image[0])
    
    def getImageHeight(self):
        return len(self.image)
        
    def getRealOutput(self,index):
        return self.realOutput[index]
    
def Treina_MLP(Mat trainingDataMat, Mat labelsMat):
    net = np.zeros((1, 3), dtype = "CV_32SC1")
    net[0][0] = 15      # entrada
    net[0][1] = N_HIDDEN # camadaOculta
    net[0][2] = N_OUTPUT # saida
    
    
    params = CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, ITERATIONS, ERROR),
                                   CvANN_MLP_TrainParams::BACKPROP,  
                                   0.1,
                                   0.1)
    
    # Treinamento MLP
    mlp = new CvANN_MLP()
    mlp.create(net, CvANN_MLP::SIGMOID_SYM, 0.6, 1)
    mlp.train(trainingDataMat, labelsMat, np.zeros(0), np.zeros(0), params)
    
    Mat resp(trainingDataMat.rows, 1, CV_32FC1)
    
    mlp->predict(trainingDataMat, resp)
    
    errados = 0
    total = 0
    predict = 0
    for i = 0 in xrange(trainingDataMat.rows):
        {
            cout << "PM = " << trainingDataMat.at<float>(i,0) << " MI = " << trainingDataMat.at<float>(i,1) << " Label: " << labelsMat.at<float>(i) << " Predict: " << resp.at<float>(i) << " Erro: " << labelsMat.at<float>(i) - resp.at<float>(i) << endl;
            
            predict = (int)(resp.at<float>(i) + 0.5);
            total++;
            if(predict != (int)(labelsMat.at<float>(i)))  
            errados++;
            }
        cout << "Total: " << total << " Acuracia: " << 100.0 * ((float) (total - errados) / (float) total) << "%" << endl;
        
        mlp->save("mlp.xml")
        
        return mlp


flag = raw_input("Escolha um opcao:\n[1] Treina a Rede\n[2] Classifica uma imagem\n[3]Analisa sequencia de movimentos")
path_dir = "/home/antonio/Documentos/SensoresInteligentes/Base/test/"
image = []

if (int(flag) == 1):
    # abre a base de dados
    for k in xrange(15):
        image_file = str(k) + ".jpg"
        fp = Images(misc.imread(path_dir + image_file, flatten=1),k)
        for i in xrange(fp.getImageHeight()):
            for j in xrange(fp.getImageWidth()):
                fp.setImageIndex(i,j,fp.getImageIndex(i,j)/255.0)
        
        image.append(fp)

    # Erro calculado
    calculatedError = 1.0

    # salva os pesos treinados em um arquivo
    data_file = "pesos.dat"
    f = open(data_file, 'wb+')

    for i in xrange(len(hiddenLayer)):
        pickle.dump(hiddenLayer[i].getWeight(),f)
        pickle.dump(hiddenLayer[i].getBiasWeight(),f)
        
    for i in xrange(len(outputLayer)):
        pickle.dump(outputLayer[i].getWeight(),f)
        pickle.dump(outputLayer[i].getBiasWeight(),f)
        
    f.close()
elif(int(flag) == 2):
    # abre a imagem que vai ser classificada e seleciona o que ela representa
    image_file = raw_input("Digite o nome da imagem: ")
    print("[Indice da Saida]\n(0 1 2 3 4)\n(5 6 7 8 9)\n(10 11 12 13 14)")
    index = raw_input("Digite o indice da saida esperada: ")
    fp = Images(misc.imread(path_dir + image_file, flatten=1),int(index))
    for i in xrange(fp.getImageHeight()):
        for j in xrange(fp.getImageWidth()):
            if (fp.getImageIndex(i,j) == 255.0):
                fp.setImageIndex(i,j,1.0)
                
    image.append(fp)
    
    # carrega os pesos previamente calculados
    data_file = "pesos.dat"
    f = open(data_file, 'rb+')

        
    f.close()
        
    # N - Taxa de Aprendizado(constante real positiva)
    N = 0.3
    
    # classifica a imagem
    calculatedError = 0.0
   
    print("Erro: " + str(calculatedError))
    print("O valor mais proximo de 1 eh a classificacao correta: ")
    print("Numero 0: " + str(outputLayer[0].getOutput()))
    print("Numero 7: " + str(outputLayer[1].getOutput()))
    print("Numero 1: " + str(outputLayer[2].getOutput()))
    print("Numero 3: " + str(outputLayer[3].getOutput()))
    print("Numero 9: " + str(outputLayer[4].getOutput()))
    print("Numero 5: " + str(outputLayer[5].getOutput()))
    print("Numero 6: " + str(outputLayer[6].getOutput()))
    print("Numero 2: " + str(outputLayer[7].getOutput()))
    print("Numero 4: " + str(outputLayer[8].getOutput()))
    print("Numero 8: " + str(outputLayer[9].getOutput()))
    print("Numero 8: " + str(outputLayer[10].getOutput()))
    print("Numero 8: " + str(outputLayer[11].getOutput()))
    print("Numero 8: " + str(outputLayer[12].getOutput()))
    print("Numero 8: " + str(outputLayer[13].getOutput()))
    print("Numero 8: " + str(outputLayer[14].getOutput()))

#cap = cv2.VideoCapture(0)
#count = 0

#while(1):
  #ret ,frame = cap.read()

  #if ret == True:
    #cv2.imshow('img2',frame)

    #k = cv2.waitKey(60) & 0xff
    #if k == 27:
      #break
    #elif k == 113:
      #cv2.imwrite(str(count)+".jpg",frame)
      #count = count + 1

  #else:
    #break

#cv2.destroyAllWindows()
#cap.release()
