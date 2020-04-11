#!/bin/bash
targetlist=`cat check_science_target_list.txt`
#targetlist='3C345'
root='./Rmag_InstMag/'
#dirlist='annu_w1_20191201-20191231'
#dirlist='annu_w1_201804-201901 annu_w1_20181101-20181130 annu_w1_201902-201910 annu_w1_20191101-20191130 annu_w1_20191201-20191231 annu_w1_20200101-20200131 annu_w1_20200201-20200229 annu_w1_20200301-20200331'
#dirlist='annu_w1_201804-201901 annu_w1_201902-201910 annu_w1_20191101-20191130 annu_w1_20191201-20191231 annu_w1_20200101-20200131 annu_w1_20200201-20200229 annu_w1_20200301-20200331'
dirlist=`ls -1 $root`
echo $dirlist

file0='tmp0.txt'
file1='tmp1.txt'
file2='tmp2.txt'
file3='tmp3.txt'
rm -rf $file0 $file1 $file2 $file3 

for i in $targetlist;do
	echo '' > $file1
	for j in $dirlist;do
		path=$root$j'/'$i'/'
		ls_file=`ls $path*dat`
		if test $ls_file; then
			echo $ls_file
			filename=`echo $ls_file|cut -d '/' -f5`
#			echo $filename
			filter=`echo $filename|cut -c6`
			cat $ls_file > $file0
			sed -i "s/$/\t$i\t$filter\t$j\/$filename/" $file0 
			cat $file0 >> $file1
		fi
	done
	cat $file1 | sort >> $file2
done


awk -v OFS="\t" '$1=$1' $file2 > $file3
sed -i "s/Lulin\t(SLT)/Lulin (SLT)/" $file3

sed -i '1 i\JulianDay\tMag\tMag_err\tObservatory\tTarget\tFilter\tPathFilename' $file3

cp $file3 gasp_target_result_join.txt
rm -rf $file0 $file1 $file2 $file3 


