all:
	clang++ main.cpp -std=c++1y -O3 -pthread

debug:
	clang++ main.cpp -std=c++1y -pthread -g3

sharc:
	g++ main.cpp -std=c++1y -O3 -fopenmp -D_GLIBCXX_PARALLEL -pthread -D N_WORKERS=12 -ltcmalloc -funroll-loops

sharc_debug:
	g++ main.cpp -std=c++1y -O0 -pthread -g3 -D N_WORKERS=1

sharc_bin:
	g++ main.cpp -std=c++1y -O3 -pthread -D N_WORKERS=12 -o bin/${FNAME} -ltcmalloc

