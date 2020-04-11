cat cata_LONEOS_2007_0.txt  | rev| cut -d "|" -f2 | rev | sort |  grep -v '^$' | wc -l
