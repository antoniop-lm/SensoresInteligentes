OPENCVFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_objdetect -lopencv_ml  -lopencv_imgcodecs
LINKERFLAGS=-I/usr/local/include -Wall -g
LINKERS=main.o
ZIP=T2SensoresInteligentes.zip Makefile *.cpp
REMOVE=*.o mlp*

all: $(LINKERS)
	@g++ $(LINKERS) $(LINKERFLAGS) -o mlp $(OPENCVFLAGS)

main.o:
	@g++ -c $(LINKERFLAGS) main.cpp $(OPENCVFLAGS)

zip:
	@zip -r $(ZIP)

clean:
	@rm -rf $(REMOVE)
