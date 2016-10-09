all:
	g++ main.cpp -std=c++1y -O3 -pthread -lopencv_core

debug:
	g++ main.cpp -std=c++1y -O3 -pthread -g3 -lopencv_core
