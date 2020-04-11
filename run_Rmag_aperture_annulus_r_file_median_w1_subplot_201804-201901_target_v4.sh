#a=`cat check_science_target_list.txt`
#a='AO0235+16'
#a='4C38-41'
#a='4C71-07'
a='ES2344+514'
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_201804-201901_target.py $i | tee python Rmag_aperture_annulus_r_file_median_w1_subplot_201804-201901_$i.log;done

