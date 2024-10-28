CWD=`pwd`
THIRDPARTY=${CWD}/thirdparty
LLAMACPP_VERSION="b3982"

# git clone -b ${LLAMACPP_VERSION} --depth 1 --recursive https://github.com/ggerganov/llama.cpp.git


get_llamacpp() {
	echo "update from llama.cpp main repo"
	PREFIX=${THIRDPARTY}/llama.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "llama.cpp" ]; then
			git clone -b ${LLAMACPP_VERSION} --depth 1 --recursive https://github.com/ggerganov/llama.cpp.git
		fi && \
		cd llama.cpp && \
		cp common/*.h ${INCLUDE} && \
		cp common/*.hpp ${INCLUDE} && \
		cp examples/llava/*.h ${INCLUDE} && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp common/libcommon.a ${LIB} && \
		cp examples/llava/libllava_static.a ${LIB}/libllava.a && \
		mv ${PREFIX}/bin ${CWD}/bin && \
		cd ${CWD}
}

get_whispercpp() {
	echo "update from whisper.cpp main repo"
	PREFIX=${THIRDPARTY}/whisper.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	BIN=${PREFIX}/bin
	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "whisper.cpp" ]; then
			git clone --depth 1 --recursive https://github.com/ggerganov/whisper.cpp.git
		fi && \
		cd whisper.cpp && \
		cp examples/*.h ${INCLUDE} && \
		cp examples/*.hpp ${INCLUDE} && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp examples/libcommon.a ${LIB} && \
		cp -rf bin/* ${BIN} && \
		cd ${CWD}
}

get_stablediffusioncpp() {
	echo "update from stable-diffusion.cpp main repo"
	PREFIX=${THIRDPARTY}/stable-diffusion.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	BIN=${PREFIX}/bin
	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "stable-diffusion.cpp" ]; then
			git clone --depth 1 --recursive https://github.com/leejet/stable-diffusion.cpp.git
		fi && \
		cd stable-diffusion.cpp && \
		cp *.h ${INCLUDE} && \
		cp *.hpp ${INCLUDE} && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp libstable-diffusion.a ${LIB} && \
		# cp -rf bin/* ${BIN} && \
		cd ${CWD}
}

get_llamacpp_python() {
	echo "update from llama-cpp-python main repo"
	PREFIX=${THIRDPARTY}/llama-cpp-python
	mkdir -p build thirdparty && \
		cd build && \
		if [ ! -d "llama-cpp-python" ]; then
			git clone --depth 1 https://github.com/abetlen/llama-cpp-python.git
		fi && \
		rm -rf ${PREFIX} && \
		mkdir -p ${PREFIX} && \
		cp -rf llama-cpp-python/llama_cpp ${PREFIX}/ && \
		cd ${CWD}
}


remove_current() {
	echo "remove current"
	rm -rf build thirdparty
}


main() {
	remove_current
	get_llamacpp
	# get_llamacpp_python
	# get_whispercpp
	# get_stablediffusioncpp
}

main


