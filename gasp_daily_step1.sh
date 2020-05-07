
file1='gasp_daily.list'
file2='gasp_daily.list2'
rm -rf $file2

a=`cat $file1`
for i in $a;do 
	date=`echo $i| cut -d / -f2 | awk -F"slt" '{print $2}'`
	fits=`echo $i| rev | cut -d / -f1 | rev`
	echo $date:$fits >> $file2
done

