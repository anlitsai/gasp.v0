a=`ls data|sort`
for i in $a;do
  echo $i
  b=`ls data/$i|sort`
  for j in $b;do
    echo $i/$j
    echo "python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log"
    python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log
  done
done
