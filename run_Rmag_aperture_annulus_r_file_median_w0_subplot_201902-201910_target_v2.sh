#obj_list='3C279 3C66A 4C29-45 Mkn501 OJ287 PKS0735+17 S4_0954+65'
obj_list='OJ287 PKS0735+17 S4_0954+65'
echo $obj_list
for i in $obj_list;do
  echo $i;
  python Rmag_aperture_annulus_r_file_median_w0_subplot_from201902_target.py $obj_name | tee Rmag_aperture_annulus_r_file_median_w0_subplot_from201902_$obj_name.log;
done

