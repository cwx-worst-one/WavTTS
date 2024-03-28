project(tagging_idl)

function(THRIFT_GEN)
  cmake_parse_arguments(thrift_gen "" "OUT_VAR;SERVER_NAME;PROTO" "" "${ARGN}")

  if(NOT thrift_gen_PROTO)
    message(SEND_ERROR "Error: no .thrift file")
    return()
  endif()

  if(NOT thrift_gen_OUT_VAR)
    message(SEND_ERROR "Error: no output variable")
    return()
  endif()

  get_filename_component(_abs_file ${thrift_gen_PROTO} ABSOLUTE)
  get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
  get_filename_component(_name ${thrift_gen_PROTO} NAME_WLE)

  file(RELATIVE_PATH _rel_dir ${CMAKE_SOURCE_DIR} ${_abs_dir})

  set(_gen_srcs)
  foreach(_ext ".cpp" ".h")
    list(APPEND _gen_srcs
         "${CMAKE_BINARY_DIR}/${_rel_dir}/${_name}_types${_ext}")
    list(APPEND _gen_srcs
         "${CMAKE_BINARY_DIR}/${_rel_dir}/${_name}_constants${_ext}")
    if(thrift_gen_SERVER_NAME)
      list(APPEND _gen_srcs
           "${CMAKE_BINARY_DIR}/${_rel_dir}/${thrift_gen_SERVER_NAME}${_ext}")
    endif()
  endforeach()

  add_custom_command(
    OUTPUT ${_gen_srcs}
    COMMAND ${CMAKE_SOURCE_DIR}/third/thrift/bin/thrift ARGS --gen cpp -out
            ${CMAKE_BINARY_DIR}/${_rel_dir}/ ${_abs_file}
    DEPENDS ${_abs_file} ${CMAKE_SOURCE_DIR}/third/thrift/bin/thrift
    COMMENT "Generating ${_gen_srcs}"
    VERBATIM)

  set_source_files_properties(${_gen_srcs} PROPERTIES GENERATED TRUE)

  if(thrift_gen_OUT_VAR)
    set(${thrift_gen_OUT_VAR}
        ${_gen_srcs}
        PARENT_SCOPE)
  else()
    message(not set)
  endif()
endfunction()

add_library(${PROJECT_NAME} STATIC)
add_library(core::idl ALIAS ${PROJECT_NAME})

function(LINK_THRIFT)
  cmake_parse_arguments(link_thrift "" "SERVER_NAME" "" "${ARGN}")
  set(_outvar)
  thrift_gen(OUT_VAR _outvar SERVER_NAME ${link_thrift_SERVER_NAME} PROTO
             ${link_thrift_UNPARSED_ARGUMENTS})
  set(_srcs)
  set(_hdr_dir)
  foreach(_file ${_outvar})
    if(_file MATCHES "cpp$")
      list(APPEND _srcs ${_file})
    else()
      get_filename_component(_hdr_dir ${_file} DIRECTORY)
    endif()
  endforeach()
  target_sources(${PROJECT_NAME} PRIVATE ${_srcs})
  target_include_directories(${PROJECT_NAME} PUBLIC ${_hdr_dir} ${_hdr_dir}/../)
endfunction()

link_thrift(base.thrift)
link_thrift(SERVER_NAME MusicTagging music_tagging.thrift)
link_thrift(SERVER_NAME InnerLangdet inner_langdet.thrift)

target_link_libraries(${PROJECT_NAME} PUBLIC deps::thrift)

