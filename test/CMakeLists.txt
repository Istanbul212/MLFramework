
enable_testing()

add_executable(matrix-ops matrix-ops.cpp)
add_test(NAME "Matrix Ops" COMMAND $<TARGET_FILE:matrix-ops>)
