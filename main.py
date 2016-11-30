from scipy import misc
import random
import numpy
import pickle

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

# Classe com os neuronios de cada camada
# weights -> pesos de entrada do neuronio
# output -> saida do neuronio
# error -> erro calculado
# v -> soma dos pesos*inputs
# bias -> valor do bias
# biasWeight -> peso do bias
# delta -> delta calculado no Backpropagation
class Neuron:
    # Flag: 
    # 1 - rede nao treinada
    # 0 - rede treinada
    def __init__(self,n_weights,flag,fl="teste"):
        if(flag):
            self.weights = []
            self.setWeights(n_weights)
            self.biasWeight = 1.0
        else:
            self.weights = pickle.load(fl)
            self.biasWeight = pickle.load(fl)
        self.output = 0.0
        self.error = 0.0
        self.v = 0.0
        self.delta = 0.0
        self.bias = 1.0
        
    # Funcao que define valores arbitrarios entre -1 e 1 para os pesos
    # n_weights -> numero de pesos
    def setWeights(self,n_weights):
        for i in xrange(n_weights):
            self.weights.append(random.uniform(-1.0, 1.0))
        
    # Funcao que calcula v e a saida gerada
    # x -> entradas
    # flag -> define se deve ser calculado para a camada escondida ou de saida
    def weightsCalculation(self,x,flag):
        self.v = 0.0
        self.v += self.bias*self.biasWeight
        if flag:
            for i in xrange(len(x)):
                aux = numpy.sum(self.weights[(i*len(x[i])):(((i+1)*len(x[i])))]*x[i])
                self.v += aux
        else:
            for i in xrange(len(x)):
                self.v += self.weights[i]*x[i].getOutput()
            
        self.output = (1 / (1 + numpy.exp(-self.v)));
        
    # Funcao que recalcula os pesos para a camada de saida
    # n -> taxa de aprendizado
    # inputs -> entradas recebidas da camada escondida
    def backpropagationOutput(self,n,inputs):
        self.delta = self.error*((1 / (1 + numpy.exp(-self.v))) * (1 - (1 / (1 + numpy.exp(-self.v)))))
        for i in xrange(len(inputs)):
            delta_peso = n*self.delta*inputs[i].getOutput()
            self.weights[i] += delta_peso
        
        self.biasWeight += n*self.delta
    
    # Funcaoque recalcula os pesos para a camada escondida
    # n -> taxa de aprendizado
    # inputs -> entradas recebidas da imagem
    # outputLayer -> camada de saida
    # index -> indice do neuronio na hidden para calculo do somatorio em delta
    def backpropagationHidden(self,n,inputs,outputLayer,index):
        sum_of_outputs = 0
        for i in xrange(len(outputLayer)):
            sum_of_outputs += outputLayer[i].backpropagationSum(index)
        
        self.delta = ((1 / (1 + numpy.exp(-self.v))) * (1 - (1 / (1 + numpy.exp(-self.v)))))*sum_of_outputs
        for i in xrange(len(inputs)):
            for j in xrange(len(inputs[0])):
                delta_peso = n*self.delta*inputs[i][j]
                self.weights[i*len(inputs[0])+j] += delta_peso
            
        self.biasWeight += n*self.delta
        
    # Funcao auxiliar para o somatorio do delta da camada escondida
    def backpropagationSum(self,index):
        return self.delta*self.weights[index]
        
    # Funcao que calcula o erro de saida
    def errorCalculation(self,training):
        self.error = training - self.output
        
    def getOutput (self):
        return self.output
    
    def getError(self):
        return self.error
    
    def getWeight(self):
        return self.weights
    
    def getBiasWeight(self):
        return self.biasWeight
    
flag = raw_input("Escolha um opcao ([1] Treina a Rede - [2] Classifica uma imagem): ")
N_HIDDEN = 45

if (int(flag) == 1):
    path_dir = "/home/antonio/Documentos/SensoresInteligentes/Base/test/"
    image = []

    # abre a base de dados
    for k in xrange(15):
        image_file = str(k) + ".jpg"
        fp = Images(misc.imread(path_dir + image_file, flatten=1),k)
        for i in xrange(fp.getImageHeight()):
            for j in xrange(fp.getImageWidth()):
                if (fp.getImageIndex(i,j) == 255.0):
                    fp.setImageIndex(i,j,1.0)
        
        image.append(fp)
        
    # inicializa as camadas
    hiddenLayer = []
    for i in xrange(N_HIDDEN):
        hiddenLayer.append(Neuron(image[0].getSize(),1))
        
    outputLayer = []
    for i in xrange(15):
        outputLayer.append(Neuron(N_HIDDEN,1))

    # Taxa de Aprendizado
    N = 0.9
    # Threshold
    error = 0.01
    # Erro calculado
    calculatedError = 1.0

    while (calculatedError >= error):
        # Para cada imagem
        calculatedError = 0.0
        for j in xrange(len(image)):
            # Forwarding
            for i in xrange(len(hiddenLayer)):
                hiddenLayer[i].weightsCalculation(image[j].getImage(),1)
            
            for i in xrange(len(outputLayer)):
                outputLayer[i].weightsCalculation(hiddenLayer,0)
                outputLayer[i].errorCalculation(image[j].getRealOutput(i))
                calculatedError += numpy.power(outputLayer[i].getError(),2)
            
            #Backpropagation
            for i in range(len(outputLayer)):
                outputLayer[i].backpropagationOutput(N,hiddenLayer)
                
            for i in range(len(hiddenLayer)):
                hiddenLayer[i].backpropagationHidden(N,image[j].getImage(),outputLayer,i)
                
        
        calculatedError /= len(image)
        print(calculatedError)

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
    path_dir = "/home/antonio/Documentos/SensoresInteligentes/Base/test/"
    image = []
    
    # abre a imagem que vai ser classificada e seleciona o que ela representa
    image_file = raw_input("Digite o nome da imagem: ")
    print("[Indice da Saida]-Numero correspondente ([0]-0, [1]-7, [2]-1, [3]-3, [4]-9, [5]-5, [6]-6, [7]-2, [8]-4, [9]-8)")
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
    hiddenLayer = []
    for i in xrange(N_HIDDEN):
        hiddenLayer.append(Neuron(image[0].getSize(),0,f))
        
    outputLayer = []
    for i in xrange(15):
        outputLayer.append(Neuron(N_HIDDEN,0,f))
        
    f.close()
        
    # N - Taxa de Aprendizado(constante real positiva)
    N = 0.3
    
    # classifica a imagem
    calculatedError = 0.0
    for i in xrange(len(hiddenLayer)):
        hiddenLayer[i].weightsCalculation(image[0].getImage(),1)
                
    for i in xrange(len(outputLayer)):
        outputLayer[i].weightsCalculation(hiddenLayer,0)
        outputLayer[i].errorCalculation(image[0].getRealOutput(i))
        calculatedError += numpy.power(outputLayer[i].getError(),2)
                    
    calculatedError /= len(image)
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
