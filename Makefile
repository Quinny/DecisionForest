all:
	g++ main.cpp -std=c++1y -O3 -pthread -lopencv_core

debug:
	g++ main.cpp -std=c++1y -O3 -pthread -g3 -lopencv_core

sharc:
	g++ main.cpp -std=c++1y -O3 -pthread -lopencv_core -I/opt/sharcnet/opencv/2.4.9/include/ -L/opt/sharcnet/opencv/2.4.9/lib/ -Wl,-R/opt/sharcnet/opencv/2.4.9/lib/ -D N_WORKERS=24

sharc_debug:
	g++ main.cpp -std=c++1y -O0 -pthread -g3 -lopencv_core -I/opt/sharcnet/opencv/2.4.9/include/ -L/opt/sharcnet/opencv/2.4.9/lib/ -Wl,-R/opt/sharcnet/opencv/2.4.9/lib/ -D N_WORKERS=24


