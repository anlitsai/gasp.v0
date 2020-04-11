a=`cat check_science_target_list.txt`
for i in $a;do python Rmag_aperture_annulus_r_file_median_w0_subplot_from201902_target.py $i | tee python Rmag_aperture_annulus_r_file_median_w0_subplot_from201902_$i.log;done

