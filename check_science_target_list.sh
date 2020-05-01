find ./ | grep GASP | grep 'fits\|fts' | cut -d / -f6 | awk -F"-2019" '{print $1}' | sort | uniq > check_science_target_list.txt

a=`find ./ | grep GASP | grep 'fits\|fts' | cut -d / -f6 | awk -F"-2019" '{print $1}' | sort | uniq | wc -l`
echo 'total science target: ' $a
echo 'check file : check_science_target_list.txt'


#3C273
#3C279
#3C345
#3C371
#3C454-3
#3C66A
#4C29-45
#4C38-41
#4C51-37
#4C71-07
#AO0235+16
#CTA102
#DA406
#ES2344+514
#KS1510-08
#L-Lacertae
#Mkn421
#Mkn501
#OJ248
#OJ287
#OJ49
#ON231
#PKS0735+17
#PKS2155-304
#S4_0954+65
#S5_0716+71