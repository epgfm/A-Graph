#! /bin/bash

find .. -type f -name "all.pdf" | xargs -II -P24 ./drop_dyn_gif.sh I
