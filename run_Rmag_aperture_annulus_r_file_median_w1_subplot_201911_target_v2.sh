#a=`cat check_science_target_list.txt`
a='4C29-45 4C71-07 AO0235+16 S4_0954+65'
d1='20191101'
d2='20191130'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_date_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done
