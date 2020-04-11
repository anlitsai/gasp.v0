#a='202003'
a=$1
for i in $a;do
  echo "python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log"
  python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log
done



