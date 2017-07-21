#!/bin/bash
FILES=predictions-*.txt
for F in $FILES
do
sed -i.bak ':a;N;$!ba;s/\n/,/g' $F
done
