all:
	g++ main.cpp -std=c++1y -O3 -pthread

debug:
	g++ main.cpp -std=c++1y -O3 -pthread -g3
