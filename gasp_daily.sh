
a=`cat gasp_daily.list`
for i in $a;do 
	date=`echo $i| cut -d / -f1 | awk -F"slt" '{print $2}'`
	fits=`echo $i| rev | cut -d / -f1 | rev`
	echo $date $fits
done

