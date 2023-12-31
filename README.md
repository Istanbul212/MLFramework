CXX=/usr/local/opt/llvm/bin/clang++
CC=/usr/local/opt/llvm/bin/clang

alias run='ASAN_OPTIONS=detect_leaks=1 ./MLFramework'
alias test='( cd test; ctest )'
