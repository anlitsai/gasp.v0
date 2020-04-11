#a=`ls data|sort`
a='202002'
for i in $a;do
  echo "python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log"
  python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log
done
