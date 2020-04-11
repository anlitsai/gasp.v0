#a=`cat check_science_target_list.txt`
a='AO0235+16'
d1='20181101'
d2='20181130'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_201811_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_201811_'$i'.log';done
#for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_201811_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done
