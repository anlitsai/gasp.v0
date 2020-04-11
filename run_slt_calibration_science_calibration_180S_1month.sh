a='202001'
b=`find ./|grep $a[0-3][0-9]$|cut -d / -f3|sort|uniq`
echo $b
for j in $b;do
  echo $j
  echo "python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log"
  python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log
done
