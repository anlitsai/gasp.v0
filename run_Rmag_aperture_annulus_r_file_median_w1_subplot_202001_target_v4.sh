#a=`cat check_science_target_list.txt`
a='CTA102'
d1='20200101'
d2='20200131'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_202001_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done
