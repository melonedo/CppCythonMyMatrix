function(Cython_add_library TARGET_NAME SOURCE_FILES DEPEND_FILES)
    set(GENERATED_FILE "${CMAKE_CURRENT_BINARY_DIR}/cython/${TARGET_NAME}.cpp")
    add_custom_command(
        OUTPUT "${GENERATED_FILE}"
        COMMAND ${Python3_EXECUTABLE} -m cython --cplus
                ${SOURCE_FILES} -o "${GENERATED_FILE}"
        DEPENDS ${SOURCE_FILES} ${DEPEND_FILES}
    )
    Python3_add_library("${TARGET_NAME}" MODULE "${GENERATED_FILE}")

    # Debug下缺少python3xx_d.lib，默认编译的话都没法调试了
    set_target_properties("${TARGET_NAME}" PROPERTIES EXCLUDE_FROM_ALL TRUE)
    if(MSVC)
        target_compile_options("${TARGET_NAME}" PRIVATE /wd4551)
    endif()
    add_custom_target("Copy${TARGET_NAME}" ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${TARGET_NAME}>" ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

function(Link_numpy TARGET_NAME)
    target_link_libraries("${TARGET_NAME}" PUBLIC Python3::NumPy)
    target_compile_definitions("${TARGET_NAME}" PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endfunction()

