# gitinit $1
# $1 is the name of the repository

git init
git remote add origin git@github.com:anlitsai/$1.git
git status
git add .
git commit -m "first commit"
gut status
git push -u origin master
