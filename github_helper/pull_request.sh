BRANCH=$(git branch | grep \* | cut -d ' ' -f2)

git checkout master
git pull
git checkout $BRANCH
git merge origin master
hub pull-request -m "$BRANCH: " -b master -e