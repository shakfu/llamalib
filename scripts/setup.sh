CWD=`pwd`
LIB=${CWD}/lib/
INCLUDE=${CWD}/include/
LLAMACPP_VERSION="bdf314f"
# git checkout bdf314f38a2c90e18285f7d7067e8d736a14000a

get_llamacpp() {
	echo "update from llama.cpp main repo"
	mkdir -p build include && \
		cd build && \
		# git clone --depth 1 --branch ${LLAMACPP_VERSION} --recursive https://github.com/ggerganov/llama.cpp.git && \
		if [ ! -d "llama.cpp" ]; then
			git clone --depth 1 --recursive https://github.com/ggerganov/llama.cpp.git
		fi && \
		# git reset --hard ${LLAMACPP_VERSION} && \
		cd llama.cpp && \
		cp common/*.h ${INCLUDE} && \
		cp common/*.hpp ${INCLUDE} && \
		cp examples/llava/*.h ${INCLUDE} && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${CWD} && \
		cp common/libcommon.a ${LIB} && \
		cp examples/llava/libllava_static.a ${LIB}/libllava.a
}

remove_current() {
	echo "remove current"
	rm -rf build/llama.cpp
	rm -rf bin include lib 
}


main() {
	remove_current && \
	get_llamacpp
}

# main
get_llamacpp
