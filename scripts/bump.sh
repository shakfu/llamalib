SHORT=`cd build/llama.cpp && git rev-parse --short HEAD`

# echo "short = ${SHORT}"

git add --all .
git commit -m "synced with llama.cpp ${SHORT}"


