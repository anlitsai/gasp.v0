#a='202003'
#a=$1
#a='201902 201903 201904 201905 201906 201907 201908 201909 201910 201911 201912 202001 202002 202003'
a='201804 201805 201806 201807 201808 201809 201810 201811 201812 201901'

c=`cat check_science_target_list.txt`

for j in $a;do

	year=`echo $j|cut -c-4`
	month=`echo $j|cut -c5-6`
	ym=`echo $year'-'$month`

	last_date=`cat gasp_target_fitsheader_info_exclude_baddata_join.txt | grep $ym|tail -1| cut -d "|" -f2| cut -d - -f3`

	d1=$j'01'
	d2=$j$last_date

	for i in $c;do python Rmag_aperture_annulus_r_file_median_w1_subplot_date_target2.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done

done
