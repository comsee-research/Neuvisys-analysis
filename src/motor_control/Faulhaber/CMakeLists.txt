cmake_minimum_required(VERSION 3.0)

project(Faulhaber)

# On inclut notre bibliothèque dans le processus de CMake
add_subdirectory(libserie)

# Crée des variables avec les fichiers à compiler
set(SRCS
        GOTO.cpp
        )

# Notre exécutable
add_executable(GOTO ${SRCS})
add_executable(MOVEOF MOVEOF.cpp)

# Et pour que l'exécutable fonctionne, il faut lui indiquer la bibliothèque dont il dépend
target_link_libraries(GOTO libserie)
target_link_libraries(MOVEOF libserie)
