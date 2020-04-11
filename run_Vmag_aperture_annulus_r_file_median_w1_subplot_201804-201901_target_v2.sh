#a=`cat check_science_target_list.txt`
#a='4C29-45 S4_0954+65'
a='S4_0954+65'
for i in $a;do python Vmag_aperture_annulus_r_file_median_w1_subplot_201804-201901_target.py $i | tee python Vmag_aperture_annulus_r_file_median_w1_subplot_201804-201901_$i.log;done

