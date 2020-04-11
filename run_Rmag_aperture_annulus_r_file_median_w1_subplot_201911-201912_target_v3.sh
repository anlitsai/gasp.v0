#a=`cat check_science_target_list.txt`
#a='3C273'
#a='3C279'
#a='4C29-45'
#a='3C66A'
#a='OJ49'
#a='S4_0954+65'
a='S5_0716+71'
d1='20191101'
d2='20191231'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_date_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done
