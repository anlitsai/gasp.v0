a=`cat dark_time_noneed.list`
echo $a
for i in $a;do find ./ | grep $i;done
b=`for i in $a;do find ./ | grep $i;done`
echo $b
rm -f $b
