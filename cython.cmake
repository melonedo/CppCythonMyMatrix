find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/MyMatrixWrapper.cpp
    COMMAND ${Python3_EXECUTABLE} -m cython --cplus
            ${CMAKE_CURRENT_SOURCE_DIR}/MyMatrixWrapper.pyx 
            -o ${CMAKE_CURRENT_BINARY_DIR}/python/MyMatrixWrapper.cpp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/MyMatrixWrapper.pyx
            ${CMAKE_CURRENT_SOURCE_DIR}/MyMatrixWrapper.pxd
)
Python3_add_library(MyMatrixWrapper MODULE
    ${CMAKE_CURRENT_BINARY_DIR}/python/MyMatrixWrapper.cpp
)
target_link_libraries(MyMatrixWrapper PUBLIC Python3::NumPy)

# Debug下缺少python3xx_d.lib，默认编译的话都没法调试了
set_target_properties(MyMatrixWrapper PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(MyMatrixWrapper PRIVATE MyMatrix)
target_compile_definitions(MyMatrixWrapper PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
if(MSVC)
	target_compile_options(MyMatrixWrapper PRIVATE /wd4551)
endif()
add_custom_target(CopyMyMatrixWrapper ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:MyMatrixWrapper> ${CMAKE_CURRENT_SOURCE_DIR})
