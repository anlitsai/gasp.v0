# gitinit $1
# $1 is the name of the repository

git init
git add .
git commit -m "first commit"
git remote add origin git@github.com:anlitsai/$1.git
git push -u origin master
