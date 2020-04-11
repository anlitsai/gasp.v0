#a=`cat check_science_target_list.txt`
#a='3C345 3C371 4C38-41 4C51-37 Mkn501 DA406 KS1510-08'
a='Mkn501'
d1='20200201'
d2='20200229'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_join_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done
