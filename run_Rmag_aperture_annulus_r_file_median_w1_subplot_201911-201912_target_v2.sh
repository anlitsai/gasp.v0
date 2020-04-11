#a=`cat check_science_target_list.txt`
#a='3C273'
#a='3C279'
#a='4C71-07'
#a='OJ287'
#a='ON231'
#a='Mkn421'
a='Mkn501'
#a='OJ248'
#a='PKS0735+17'
d1='20191101'
d2='20191231'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_date_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done
