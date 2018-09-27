CC = g++
LIB_INPUTS = ./include/Matrix.cpp ./include/NeuralNetwork.cpp
LIB_OUTPUTS = Matrix.o NeuralNetwork.o
EXPORT_FLAG = TESTLIBRARY_EXPORTS
EXTRA_FLAGS = -std=c++11 -Wall -Os -fPIC
LIB_NAME = NeuralNetwork
STATIC_EXTENSION = lib #a
DYNAMIC_EXTENSION = dll #so
INCLUDE_PATH = ./include

#EXE_FILES = test.cpp $(LIB_NAME).$(DYNAMIC_EXTENSION) 
#OUT_EXE = tester.exe

rebuild: clean build clean_o

lib :
	$(CC) $(EXTRA_FLAGS) -c $(LIB_INPUTS) -D$(EXPORT_FLAG) #-o $(LIB_OUTPUTS)
	ar rcs $(LIB_NAME).$(STATIC_EXTENSION) $(LIB_OUTPUTS)
	$(CC) $(EXTRA_FLAGS) -shared -o $(LIB_NAME).$(DYNAMIC_EXTENSION) $(LIB_OUTPUTS)
	echo Lib done

#exe : lib $(EXE_FILES)
#	$(CC) $(EXTRA_FLAGS) -I$(INCLUDE_PATH) -o $(OUT_EXE) $(EXE_FILES)
#	echo Exe done

build : lib #exe

clean : 
	-rm -f $(LIB_OUTPUTS)
	-rm -f $(LIB_NAME).$(STATIC_EXTENSION) $(LIB_NAME).$(DYNAMIC_EXTENSION)
	#-rm -f $(OUT_EXE)
	echo Clean done

clean_o :
	-rm -f $(LIB_OUTPUTS)