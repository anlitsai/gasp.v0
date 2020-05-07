sourcelist=`cat check_science_target_list.txt`
filelist=`cat gasp_daily.list2`

file3='gasp_daily.list3'
file4='gasp_daily.list4'
rm -rf $file3 $file4

for i in $filelist;do
	date=`echo $i|cut -d ":" -f1`
	fits=`echo $i|cut -d ":" -f2`
#	echo $date $fits
	for source in $sourcelist;do
		if grep -q $source <<< $fits;then
			echo $date $source >> $file3
		fi
	done
done

cat $file3 | sort | uniq > $file4
		
