BRANCH=$(git branch | grep \* | cut -d ' ' -f2)

git checkout master
git pull
git checkout $BRANCH
git merge origin master
if [ ! "$1" = "--no-edit" ];
then
  hub pull-request -m "$BRANCH: $1" -b master -e
else
  hub pull-request -b master --no-edit
fi