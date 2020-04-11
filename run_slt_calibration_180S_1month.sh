a='201912'
for i in $a;do
  echo "python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log"
  python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log
done


b=`find ./|grep $a[0-3][0-9]$|cut -d / -f3|sort|uniq`
echo $b
for j in $b;do
  echo $j
  echo "python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log"
  python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log
done


python gasp_target_fitsheader_info_exclude_baddata_201911-201912.py


c=`cat check_science_target_list.txt`
d1='20191101'
d2='20191231'
for i in $c;do python Rmag_aperture_annulus_r_file_median_w1_subplot_date_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done


