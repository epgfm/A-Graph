#! /bin/bash


echo "Run Start"

find .. -type d -name "dyn" | while read d; do

    echo "target dir:" $d
    cd $d
    TARGETS=`ls *.pdf | sort -n | xargs echo`
    # Concatenate target pdf files
    pdftk $TARGETS cat output all.pdf
    rm -f $TARGETS
    cd -

done

echo "End Run"


