a=`cat check_science_target_list.txt`
for i in $a;do python Rmag_aperture_annulus_r_file_median_w1_subplot_201909-201910_target.py $i | tee python Rmag_aperture_annulus_r_file_median_w1_subplot_201909-201910_$i.log;done

