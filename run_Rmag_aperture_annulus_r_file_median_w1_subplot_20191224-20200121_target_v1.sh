a=`cat check_science_target_list.txt`
d1='20191224'
d2='20200121'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_join_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done
