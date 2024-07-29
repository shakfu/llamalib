CWD=`pwd`
LLAMACPP_VERSION="0832de7"

get_llamacpp() {
	echo "update from llama.cpp main repo"
	mkdir -p build && \
		cd build && \
		# git clone --depth 1 --branch ${LLAMACPP_VERSION} --recursive https://github.com/ggerganov/llama.cpp.git && \
		git clone --depth 1 --recursive https://github.com/ggerganov/llama.cpp.git && \
		# git reset --hard ${LLAMACPP_VERSION} && \
		cd llama.cpp && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${CWD}
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

main

