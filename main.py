from scipy import misc
import random
import numpy as np
import cv2
import pickle

N_INPUT = 307200
N_HIDDEN = 45
N_OUTPUT = 15
ITERATIONS = 6000
ERROR = 0.01

class Classifica:
    def __init__(self,mlp):
        self.mlp = mlp
        
    def predict(self,inputs,labels):
        self.mlp.predict(inputs,labels)
            
    def predictRound(self,inputs,labels):
        self.predict(inputs, labels);
        for i in xrange(labels.rows):
            for j in xrange(labels.cols):
                labels[j][i] = float(int(labels[j][i] + 0.5))
                

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
    
def Treina_MLP(self,trainingDataMat,labelsMat):
    net = np.zeros((1, 3), dtype = "CV_32SC1")
    net[0][0] = N_INPUT     # entrada
    net[0][1] = N_HIDDEN    # camadaOculta
    net[0][2] = N_OUTPUT    # saida

    params = cv2.CvANN_MLP_TrainParams(cv2.cvTermCriteria(cv2.CV_TERMCRIT_ITER + cv2.CV_TERMCRIT_EPS, ITERATIONS, ERROR), cv2.CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1)
    
    # Treinamento MLP
    mlp = cv2.CvANN_MLP()
    mlp.create(net, cv2.CvANN_MLP::SIGMOID_SYM, 0.6, 1)
    mlp.train(trainingDataMat, labelsMat, np.zeros(0), np.zeros(0), params)
    
    mlp.save("mlp.xml")

def Testa_MLP(self,trainingDataMat,labelsMat):
    mlp = cv2.CvANN_MLP()
    mlp.load("mlp.xml")
    # OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #resp(trainingDataMat.rows, 1, CV_32FC1)
    # OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    mlp.predict(trainingDataMat, resp)
    
    errados = 0
    total = 0
    predict = 0
    for i = 0 in xrange(trainingDataMat.rows):
        print("PM = " + str(trainingDataMat[i][0]) + " MI = " + str(trainingDataMat[i][1]) + " Label: " + str(labelsMat[i]) + " Predict: " + str(resp[i]) + " Erro: " + str(labelsMat[i] - resp[i]))
        
        predict = int(resp[i] + 0.5)
        total = total + 1
        if(predict != int(labelsMat[i])):  
            errados = errados + 1
    
    print("Total: " + str(total) + " Acuracia: " + str(100.0 * ((total - errados) / total)) + "%")
    
def CompareResult(self,actualResult,lastResult):
    print("em andamento")

flag = raw_input("Escolha um opcao:\n[1] Treina a Rede\n[2] Classifica uma imagem\n[3]Analisa sequencia de movimentos")
path_dir = "/home/antonio/Documentos/SensoresInteligentes/Base/test/"

if (int(flag) == 1):
    trainingDataMat = np.zeros((15, N_INPUT))
    labelsMat = np.zeros((15,15))
    # abre a base de dados
    for k in xrange(15):
        image_file = str(k) + ".jpg"
        fp = Images(misc.imread(path_dir + image_file, flatten=1),k)
        for i in xrange(fp.getImageHeight()):
            for j in xrange(fp.getImageWidth()):
                trainingDataMat[k][i*fp.getImageWidth()+j] = fp.getImageIndex(i,j)/255.0
        
        labelsMat[k][k] = 1
        
    Treina_MLP(trainingDataMat,labelsMat)
elif(int(flag) == 2):
    trainingDataMat = np.zeros((1, N_INPUT))
    labelsMat = np.zeros((1,15))
    # abre a imagem que vai ser classificada e seleciona o que ela representa
    image_file = raw_input("Digite o nome da imagem: ")
    print("[Indice da Saida]\n(0 1 2 3 4)\n(5 6 7 8 9)\n(10 11 12 13 14)")
    index = raw_input("Digite o indice da saida esperada: ")
    fp = Images(misc.imread(path_dir + image_file, flatten=1),int(index))
    for i in xrange(fp.getImageHeight()):
        for j in xrange(fp.getImageWidth()):
            trainingDataMat[0][i*fp.getImageWidth()+j] = fp.getImageIndex(i,j)/255.0
                
    labelsMat[0][int(index)] = 1
    
    Testa_MLP(trainingDataMat,labelsMat)
elif(int(flag) == 3):
    trainingDataMat = np.zeros((1, N_INPUT))
    labelsMat = np.zeros((1,15))
    lastResult = np.zeros((1,15))
    actualResult = np.zeros((1,15))
    cap = cv2.VideoCapture(0)

    while(1):
        ret,frame = cap.read()

        if ret == True:
            cv2.imshow('img2',frame)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i in xrange(cap.get(4)):
                for j in xrange(cap.get(3)):
                    trainingDataMat[0][i*cap.get(3)+j] = frame[i][j]/255.0
            
            actualResult = Testa_MLP(trainingDataMat,labelsMat)
            CompareResult(actualResult,lastResult)
            lastResult = actualResult

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break

        else:
            break

    cv2.destroyAllWindows()
    cap.release()
