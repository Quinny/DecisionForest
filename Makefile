all:
	clang++ main.cpp -std=c++1y -O3 -pthread

debug:
	clang++ main.cpp -std=c++1y -O3 -pthread -g3

sharc:
	g++ main.cpp -std=c++1y -O3 -pthread -D N_WORKERS=12 -ltcmalloc

sharc_debug:
	g++ main.cpp -std=c++1y -O0 -pthread -g3 -D N_WORKERS=12

sharc_bin:
	g++ main.cpp -std=c++1y -O3 -pthread -D N_WORKERS=12 -o bin/${FNAME} -ltcmalloc

