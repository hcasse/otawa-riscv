cmake_minimum_required(VERSION 2.8)

# MIPS plugin build
set(ISA 			"riscv")
set(NAMESPACE		"otawa")
set(ELF_NUM			"243")		 # 0xf3
set(CONFIG			)

# looking for ARM loader
set(ISA_PATH "${CMAKE_SOURCE_DIR}/../${ISA}")
find_library(ISA_LIB "${ISA}" PATHS "${ISA_PATH}/src" NO_DEFAULT_PATH)
if(ISA_LIB)
	message(STATUS "${ISA} library found at ${ISA_LIB}")
else()
	message(FATAL_ERROR "${ISA} library not found")
endif()
include_directories("${ISA_PATH}/include")

set(ISA_PREFIX 	"${ISA_PATH}")
set(ISA_LIB 	"${ISA_PREFIX}/src/lib${ISA}.a")
set(ISA_IRG		"${ISA_PREFIX}/${ISA}.irg")


# look for OTAWA
if(NOT OTAWA_CONFIG)
    find_program(OTAWA_CONFIG otawa-config DOC "path to otawa-config")
    if(NOT OTAWA_CONFIG)
        message(FATAL_ERROR "ERROR: otawa-config is required !")
    endif()
endif()
message(STATUS "otawa-config found at ${OTAWA_CONFIG}")


# get OTAWA configuration
execute_process(COMMAND "${OTAWA_CONFIG}" --cflags
	OUTPUT_VARIABLE OTAWA_CFLAGS  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${OTAWA_CONFIG}" --libs
	OUTPUT_VARIABLE OTAWA_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${OTAWA_CONFIG}" --prefix --rpath
	OUTPUT_VARIABLE OTAWA_PREFIX  OUTPUT_STRIP_TRAILING_WHITESPACE)
include("${OTAWA_PREFIX}/share/Otawa/cmake/Otawa.cmake")
set(CMAKE_CXX_STANDARD 11)

add_compile_options(-Wall)
if(CMAKE_VERSION LESS "3.1")
	add_compile_options("--std=c++11")
	message(STATUS "C++11 set using cflags")
else()
	set(CMAKE_CXX_STANDARD 11)
	message(STATUS "C++ set using CMAKE_CXX_STANDARD")
endif()


# sources
set(GLISS_ATTR		"${CMAKE_SOURCE_DIR}/../gliss2/gep/gliss-attr")
MAKE_GLISS_FUNCTION(OTAWA_KIND    "otawa_kind"      "otawa_kind.nmp"
	"${ISA_IRG}"    "return 0;")
MAKE_GLISS_FUNCTION(OTAWA_TARGET  "otawa_target"    "otawa_target.nmp"
	"${ISA_IRG}"  "return 0;")
MAKE_GLISS_FUNCTION(OTAWA_DELAYED "otawa_delayed"   "otawa_target.nmp"
	"${ISA_IRG}"   "return 0;")
MAKE_GLISS_PROCEDURE(OTAWA_READ  "otawa_read"   "otawa_uregs.nmp"
	"${ISA_IRG}"   ";")
MAKE_GLISS_PROCEDURE(OTAWA_WRITE  "otawa_write"   "otawa_uregs.nmp"
	"${ISA_IRG}"   ";")
#MAKE_GLISS_PROCEDURE(OTAWA_SEM    "otawa_sem"       "otawa_sem.nmp"
#	"${PROC_IRG}"      ";"           "${ARM_AUX}")

set(SOURCES
	"${ISA}.cpp"
	${OTAWA_KIND}
	${OTAWA_TARGET}
	${OTAWA_DELAYED}
	${OTAWA_READ}
	${OTAWA_WRITE}
)


# build the library
file(STRINGS VERSION VERSION LIMIT_COUNT 1)
set(CMAKE_CXX_FLAGS "-Wall -DISA_VERSION=\"\\\"${VERSION}\\\"\"" )
set(ORIGIN $ORIGIN)
add_library("${ISA}" SHARED ${SOURCES})
set_property(TARGET "${ISA}" PROPERTY PREFIX "")
set_property(TARGET "${ISA}" PROPERTY COMPILE_FLAGS "${OTAWA_CFLAGS}")
target_link_libraries("${ISA}" "${OTAWA_LDFLAGS} ${ISA_LIB}")


# installation
set(PLUGIN_PATH "${OTAWA_PREFIX}/lib/otawa/${NAMESPACE}")
install(TARGETS "${ISA}" LIBRARY		DESTINATION "${PLUGIN_PATH}")
install(FILES	"${ISA}.eld"			DESTINATION "${PLUGIN_PATH}")
install(FILES	"elf_${ELF_NUM}.eld"	DESTINATION "${OTAWA_PREFIX}/lib/otawa/loader")
if(CONFIG)
	install(FILES	"${CONFIG}"			DESTINATION "${PLUGIN_PATH}")
endif()
