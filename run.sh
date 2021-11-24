#!/bin/zsh

rm -f core
cd ./build
cmake .. > /dev/null
make -j > /dev/null 2> /dev/null

compile=$?

cd ..;

if [ $compile -ne 0 ]; then
    echo 'Compile error'
    return
fi

echo "[SHELL] start..."
mpirun --allow-run-as-root --mca btl tcp,self --hostfile ./hosts -n 2 ./build/main
